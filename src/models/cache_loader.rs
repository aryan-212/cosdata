use super::buffered_io::{BufIoError, BufferManagerFactory};
use super::common::TSHashTable;
use super::file_persist::read_prop_from_file;
use super::lazy_load::{FileIndex, LazyItem, LazyItemVec, VectorData};
use super::lru_cache::LRUCache;
use super::prob_lazy_load::lazy_item::{ProbLazyItem, ProbLazyItemState, ReadyState};
use super::prob_node::{ProbNode, SharedNode};
use super::serializer::prob::ProbSerialize;
use super::serializer::CustomSerialize;
use super::types::*;
use super::versioning::Hash;
use crate::config_loader::Config;
use crate::models::common::*;
use crate::models::lru_cache::CachedValue;
use crate::storage::inverted_index_old::InvertedIndexItem;
use crate::storage::inverted_index_sparse_ann::{
    InvertedIndexSparseAnn, InvertedIndexSparseAnnNode,
};
use crate::storage::inverted_index_sparse_ann_basic::InvertedIndexSparseAnnNodeBasic;
use crate::storage::inverted_index_sparse_ann_basic::{
    InvertedIndexSparseAnnNodeBasicDashMap, InvertedIndexSparseAnnNodeBasicTSHashmap,
};
use crate::storage::inverted_index_sparse_ann_new_ds::InvertedIndexNewDSNode;
use crate::storage::Storage;
use arcshift::ArcShift;
use dashmap::DashMap;
use probabilistic_collections::cuckoo::CuckooFilter;
use std::collections::HashSet;
use std::fs::File;
use std::io;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::TryLockError;
use std::sync::{atomic::AtomicBool, Arc, Mutex, RwLock, Weak};

use crate::models::lazy_load::SyncPersist;

macro_rules! define_cache_items {
    ($($variant:ident = $type:ty),+ $(,)?) => {
        #[derive(Clone)]
        pub enum CacheItem {
            $($variant(LazyItem<$type>)),+
        }


        pub trait Cacheable: Clone + 'static {
            fn from_cache_item(cache_item: CacheItem) -> Option<LazyItem<Self>>;
            fn into_cache_item(item: LazyItem<Self>) -> CacheItem;
        }

        $(
            impl Cacheable for $type {
                fn from_cache_item(cache_item: CacheItem) -> Option<LazyItem<Self>> {
                    if let CacheItem::$variant(item) = cache_item {
                        Some(item)
                    } else {
                        None
                    }
                }

                fn into_cache_item(item: LazyItem<Self>) -> CacheItem {
                    CacheItem::$variant(item)
                }
            }
        )+
    };
}

define_cache_items! {
    MergedNode = MergedNode,
    Storage = Storage,
    InvertedIndexItemWithStorage = InvertedIndexItem<Storage>,
    Float = f32,
    Unsigned32 = u32,
    InvertedIndexItemWithFloat = InvertedIndexItem<f32>,
    InvertedIndexSparseAnnNode = InvertedIndexSparseAnnNode,
    InvertedIndexSparseAnnNodeBasic = InvertedIndexSparseAnnNodeBasic,
    InvertedIndexSparseAnn = InvertedIndexSparseAnn,
    // InvertedIndexSparseAnnNodeBasicTSHashmap = InvertedIndexSparseAnnNodeBasicTSHashmap,
    InvertedIndexSparseAnnNodeBasicDashMap = InvertedIndexSparseAnnNodeBasicDashMap,
    InvertedIndexNewDSNode = InvertedIndexNewDSNode,
    VectorData = STM<VectorData>,
}

pub struct NodeRegistry {
    cuckoo_filter: RwLock<CuckooFilter<u64>>,
    registry: LRUCache<u64, CacheItem>,
    bufmans: Arc<BufferManagerFactory<Hash>>,
}

impl NodeRegistry {
    pub fn new(cuckoo_filter_capacity: usize, bufmans: Arc<BufferManagerFactory<Hash>>) -> Self {
        let cuckoo_filter = CuckooFilter::new(cuckoo_filter_capacity);
        let registry = LRUCache::with_prob_eviction(1000, 0.03125);
        NodeRegistry {
            cuckoo_filter: RwLock::new(cuckoo_filter),
            registry,
            bufmans,
        }
    }

    pub fn get_bufmans(&self) -> Arc<BufferManagerFactory<Hash>> {
        self.bufmans.clone()
    }

    pub fn get_object<T: Cacheable, F>(
        self: Arc<Self>,
        file_index: FileIndex,
        load_function: F,
        max_loads: u16,
        skipm: &mut HashSet<u64>,
    ) -> Result<LazyItem<T>, BufIoError>
    where
        F: Fn(
            Arc<BufferManagerFactory<Hash>>,
            FileIndex,
            Arc<Self>,
            u16,
            &mut HashSet<u64>,
        ) -> Result<LazyItem<T>, BufIoError>,
    {
        println!(
            "get_object called with file_index: {:?}, max_loads: {}",
            file_index, max_loads
        );

        let combined_index = Self::combine_index(&file_index);

        {
            let cuckoo_filter = self.cuckoo_filter.read().unwrap();
            println!("Acquired read lock on cuckoo_filter");

            // Initial check with Cuckoo filter
            if cuckoo_filter.contains(&combined_index) {
                println!("FileIndex found in cuckoo_filter");
                if let Some(obj) = self.registry.get(&combined_index) {
                    if let Some(item) = T::from_cache_item(obj) {
                        println!("Object found in registry, returning");
                        return Ok(item);
                    }
                } else {
                    println!("Object not found in registry despite being in cuckoo_filter");
                }
            } else {
                println!("FileIndex not found in cuckoo_filter");
            }
        }
        println!("Released read lock on cuckoo_filter");

        let (version_id, version_number) = if let FileIndex::Valid {
            version_id,
            version_number,
            ..
        } = &file_index
        {
            (*version_id, *version_number)
        } else {
            (0.into(), 0)
        };

        if max_loads == 0 || !skipm.insert(combined_index) {
            println!("Either max_loads hit 0 or loop detected, returning LazyItem with no data");
            return Ok(LazyItem::Valid {
                data: ArcShift::new(None),
                file_index: ArcShift::new(Some(file_index)),
                decay_counter: 0,
                persist_flag: Arc::new(AtomicBool::new(true)),
                versions: LazyItemVec::new(),
                version_id,
                version_number,
                serialized_flag: Arc::new(AtomicBool::new(true)),
            });
        }

        println!("Calling load_function");
        let item = load_function(
            self.bufmans.clone(),
            file_index.clone(),
            self.clone(),
            max_loads - 1,
            skipm,
        )?;
        println!("load_function returned successfully");

        println!("Trying to get or insert item into registry");
        let cached_item = self
            .registry
            .get_or_insert::<BufIoError>(combined_index.clone(), || Ok(T::into_cache_item(item)))?;

        match cached_item {
            CachedValue::Hit(item) => {
                println!("Object found in registry after load, returning");
                Ok(T::from_cache_item(item).unwrap())
            }
            CachedValue::Miss(item) => {
                println!("Inserting key into cuckoo_filter");
                self.cuckoo_filter.write().unwrap().insert(&combined_index);

                println!("Returning newly created LazyItem");
                Ok(T::from_cache_item(item).unwrap())
            }
        }
    }

    pub fn load_item<T: CustomSerialize>(
        self: Arc<Self>,
        file_index: FileIndex,
    ) -> Result<T, BufIoError> {
        let mut skipm: HashSet<u64> = HashSet::new();

        if file_index == FileIndex::Invalid {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Cannot deserialize with an invalid FileIndex",
            )
            .into());
        };

        T::deserialize(
            self.bufmans.clone(),
            file_index,
            self.clone(),
            1000,
            &mut skipm,
        )
    }

    pub fn combine_index(file_index: &FileIndex) -> u64 {
        match file_index {
            FileIndex::Valid {
                offset, version_id, ..
            } => ((offset.0 as u64) << 32) | (**version_id as u64),
            FileIndex::Invalid => u64::MAX, // Use max u64 value for Invalid
        }
    }

    // pub fn split_combined_index(combined: u64) -> FileIndex {
    //     if combined == u64::MAX {
    //         FileIndex::Invalid
    //     } else {
    //         FileIndex::Valid {
    //             offset: FileOffset((combined >> 32) as u32),
    //             version: (combined as u32).into(),
    //         }
    //     }
    // }
}

macro_rules! define_prob_cache_items {
    ($($variant:ident = $type:ty),+ $(,)?) => {
        #[derive(Clone)]
        pub enum ProbCacheItem {
            $($variant(*mut ProbLazyItem<$type>)),+
        }


        pub trait ProbCacheable: 'static + Sized {
            fn from_cache_item(cache_item: ProbCacheItem) -> Option<*mut ProbLazyItem<Self>>;
            fn into_cache_item(item: *mut ProbLazyItem<Self>) -> ProbCacheItem;
        }

        $(
            impl ProbCacheable for $type {
                fn from_cache_item(cache_item: ProbCacheItem) -> Option<*mut ProbLazyItem<Self>> {
                    if let ProbCacheItem::$variant(item) = cache_item {
                        Some(item)
                    } else {
                        None
                    }
                }

                fn into_cache_item(item: *mut ProbLazyItem<Self>) -> ProbCacheItem {
                    ProbCacheItem::$variant(item)
                }
            }
        )+
    };
}

define_prob_cache_items! {
    ProbNode = ProbNode,
    InvertedIndex = InvertedIndexSparseAnnNodeBasicTSHashmap
}

pub struct ClusterCache {
    // Cache of entire clusters
    cluster_registry: LRUCache<u64, Arc<Cluster>>,
    cluster_roots: TSHashTable<u64, Arc<Cluster>>,
    loading_items: TSHashTable<Hash, Arc<Mutex<bool>>>,
    batch_load_lock: Mutex<()>,
}

#[derive(Clone)]
pub struct Cluster {
    pub id: Hash,               // Unique identifier for the cluster
    pub root_node: SharedNode,  // Root node of this cluster's HNSW
    pub size: Arc<AtomicUsize>, // Current size of the cluster
    // File managers specific to this cluster
    pub index_manager: Arc<BufferManagerFactory<Hash>>,
    // Cache specific to this cluster
    pub cache: Arc<ProbCache>,
    pub items: Arc<TSHashTable<u64, ProbCacheItem>>,
}

impl Cluster {
    pub fn new(
        id: Hash,
        root_node: SharedNode,
        index_path: Arc<Path>,
        config: &Config,
        prop_file: Arc<RwLock<File>>,
    ) -> Result<Self, WaCustomError> {
        let index_manager = Arc::new(BufferManagerFactory::new(
            index_path.clone(),
            |root, ver: &Hash| root.join(format!("{}.chn", **ver)),
            config.flush_eagerness_factor,
            ProbNode::get_serialized_size(config.hnsw.default_neighbors_count) * 1000,
        ));

        let cache = Arc::new(ProbCache::new(
            index_manager.clone(),
            index_manager.clone(),
            prop_file,
        ));
        let items = Arc::new(TSHashTable::new(8));

        Ok(Self {
            id,
            root_node,
            size: Arc::new(AtomicUsize::new(1)), // Start with 1 (root node)
            index_manager,
            cache,
            items,
        })
    }

    pub fn increment_size(&self) {
        self.size.fetch_add(1, Ordering::SeqCst);
    }

    pub fn get_size(&self) -> usize {
        self.size.load(Ordering::SeqCst)
    }
}

pub fn create_new_cluster(
    cluster_cache: Arc<ClusterCache>,
    root_node: SharedNode,
    index_path: Arc<Path>,
    config: &Config,
    prop_file: Arc<RwLock<File>>,
) -> Result<Arc<Cluster>, WaCustomError> {
    let cluster_id = unsafe { &*root_node }.get_current_version();

    let index_manager = Arc::new(BufferManagerFactory::new(
        index_path,
        |root, ver: &Hash| root.join(format!("{}.chn", **ver)),
        config.flush_eagerness_factor,
        ProbNode::get_serialized_size(config.hnsw.default_neighbors_count) * 1000,
    ));

    let cache = Arc::new(ProbCache::new(
        index_manager.clone(),
        index_manager.clone(),
        prop_file,
    ));

    let items = Arc::new(TSHashTable::new(16));

    let cluster = Arc::new(Cluster {
        id: cluster_id,
        root_node,
        size: Arc::new(AtomicUsize::new(1)),
        index_manager,
        cache,
        items,
    });

    // Set the cluster reference and root flag in the root node
    let node_id = unsafe { &*root_node }
        .try_get_data(&cluster.cache)?
        .prop
        .id
        .0;
    cluster_cache.cluster_roots.insert(node_id, cluster.clone());

    cluster_cache
        .cluster_registry
        .insert(*cluster_id as u64, cluster.clone());

    Ok(cluster)
}

pub struct ProbCache {
    registry: LRUCache<u64, ProbCacheItem>,
    props_registry: DashMap<u64, Weak<NodeProp>>,
    bufmans: Arc<BufferManagerFactory<Hash>>,
    level_0_bufmans: Arc<BufferManagerFactory<Hash>>,
    prop_file: Arc<RwLock<File>>,
    loading_items: TSHashTable<u64, Arc<Mutex<bool>>>,
    // A global lock to prevent deadlocks during batch loading of cache entries when `max_loads > 1`.
    //
    // This lock ensures that only one thread is allowed to load large batches of nodes (where `max_loads > 1`)
    // at any given time. If multiple threads attempt to load interconnected nodes in parallel with high `max_loads`,
    // it can lead to a deadlock situation due to circular dependencies between the locks. By serializing access to
    // large batch loads, this mutex ensures that only one thread can initiate a batch load with a high `max_loads`
    // value, preventing such circular waiting conditions. Threads with `max_loads = 1` can still load nodes in parallel
    // without causing conflicts, allowing for efficient loading of smaller batches.
    batch_load_lock: Mutex<()>,
}

unsafe impl Send for ProbCache {}
unsafe impl Sync for ProbCache {}

impl ProbCache {
    pub fn new(
        bufmans: Arc<BufferManagerFactory<Hash>>,
        level_0_bufmans: Arc<BufferManagerFactory<Hash>>,
        prop_file: Arc<RwLock<File>>,
    ) -> Self {
        let registry = LRUCache::with_prob_eviction(100_000_000, 0.03125);
        let props_registry = DashMap::new();

        Self {
            registry,
            props_registry,
            bufmans,
            level_0_bufmans,
            prop_file,
            loading_items: TSHashTable::new(16),
            batch_load_lock: Mutex::new(()),
        }
    }

    pub fn get_prop(
        &self,
        offset: FileOffset,
        length: BytesToRead,
    ) -> Result<Arc<NodeProp>, BufIoError> {
        let key = Self::get_prop_key(offset, length);
        if let Some(prop) = self
            .props_registry
            .get(&key)
            .and_then(|prop| prop.upgrade())
        {
            return Ok(prop);
        }
        let mut prop_file_guard = self.prop_file.write().unwrap();
        let prop = Arc::new(read_prop_from_file(
            (offset, length),
            &mut *prop_file_guard,
        )?);
        drop(prop_file_guard);
        let weak = Arc::downgrade(&prop);
        self.props_registry.insert(key, weak);
        Ok(prop)
    }

    pub fn insert_lazy_object(&self, version: Hash, offset: u32, item: SharedNode) {
        let combined_index = (offset as u64) << 32 | (*version as u64);
        if let Some(node) = unsafe { &*item }.get_lazy_data() {
            let prop_key = Self::get_prop_key(node.prop.location.0, node.prop.location.1);
            self.props_registry
                .insert(prop_key, Arc::downgrade(&node.prop));
        }
        self.registry
            .insert(combined_index, ProbNode::into_cache_item(item));
    }

    pub fn force_load_single_object<T: ProbSerialize + ProbCacheable>(
        &self,
        file_index: FileIndex,
        is_level_0: bool,
    ) -> Result<*mut ProbLazyItem<T>, BufIoError> {
        let combined_index = Self::combine_index(&file_index, is_level_0);
        let mut skipm = HashSet::new();
        skipm.insert(combined_index);
        let data = T::deserialize(
            &self.bufmans,
            &self.level_0_bufmans,
            file_index,
            self,
            0,
            &mut skipm,
            is_level_0,
        )?;
        let (file_offset, version_number, version_id) = match file_index {
            FileIndex::Valid {
                offset,
                version_number,
                version_id,
            } => (offset, version_number, version_id),
            FileIndex::Invalid => unreachable!(),
        };
        let state = ProbLazyItemState::Ready(ReadyState {
            data,
            file_offset,
            is_serialized: AtomicBool::new(true),
            persist_flag: AtomicBool::new(false),
            version_id,
            version_number,
        });

        let item = ProbLazyItem::new_from_state(state, is_level_0);

        self.registry
            .insert(combined_index.clone(), T::into_cache_item(item.clone()));

        Ok(item)
    }

    pub fn get_lazy_object<T: ProbSerialize + ProbCacheable>(
        &self,
        file_index: FileIndex,
        max_loads: u16,
        skipm: &mut HashSet<u64>,
        is_level_0: bool,
    ) -> Result<*mut ProbLazyItem<T>, BufIoError> {
        let combined_index = Self::combine_index(&file_index, is_level_0);

        if let Some(item) = self.registry.get(&combined_index) {
            if let Some(item) = T::from_cache_item(item) {
                return Ok(item);
            }
        }

        if max_loads == 0 || !skipm.insert(combined_index) {
            return Ok(ProbLazyItem::new_pending(file_index, is_level_0));
        }

        let mut mutex = self
            .loading_items
            .get_or_create(combined_index, || Arc::new(Mutex::new(false)));
        let mut load_complete = mutex.lock().unwrap();

        loop {
            // check again
            if let Some(item) = self.registry.get(&combined_index) {
                if let Some(item) = T::from_cache_item(item) {
                    return Ok(item);
                }
            }

            // another thread loaded the data but its not in the registry (got evicted), retry
            if *load_complete {
                drop(load_complete);
                mutex = self
                    .loading_items
                    .get_or_create(combined_index, || Arc::new(Mutex::new(false)));
                load_complete = mutex.lock().unwrap();
                continue;
            }

            break;
        }

        let (file_offset, version_number, version_id) = if let FileIndex::Valid {
            offset,
            version_number,
            version_id,
        } = &file_index
        {
            (*offset, *version_number, *version_id)
        } else {
            (FileOffset(0), 0, 0.into())
        };

        let data = T::deserialize(
            &self.bufmans,
            &self.level_0_bufmans,
            file_index,
            self,
            max_loads - 1,
            skipm,
            is_level_0,
        )?;
        let state = ProbLazyItemState::Ready(ReadyState {
            data,
            file_offset,
            is_serialized: AtomicBool::new(true),
            persist_flag: AtomicBool::new(false),
            version_id,
            version_number,
        });

        let item = ProbLazyItem::new_from_state(state, is_level_0);

        self.registry
            .insert(combined_index.clone(), T::into_cache_item(item.clone()));

        *load_complete = true;
        self.loading_items.delete(&combined_index);

        Ok(item)
    }

    pub fn load_region(
        &self,
        region_start: u32,
        version_number: u16,
        version_id: Hash,
        node_size: u32,
        is_level_0: bool,
    ) -> Result<Vec<SharedNode>, BufIoError> {
        let bufman = if is_level_0 {
            self.level_0_bufmans.get(version_id)?
        } else {
            self.bufmans.get(version_id)?
        };
        let file_size = bufman.file_size();
        if region_start as u64 > file_size {
            return Ok(Vec::new());
        }
        println!(
            "Loading region: {}, version: {}, is_level_0: {}",
            region_start, version_number, is_level_0
        );
        let cap = ((file_size - region_start as u64) / node_size as u64).min(1000) as usize;
        let mut nodes = Vec::with_capacity(cap);
        for i in 0..1000 {
            let offset = FileOffset(i * node_size + region_start);
            if offset.0 as u64 >= file_size {
                break;
            }
            let file_index = FileIndex::Valid {
                offset,
                version_number,
                version_id,
            };
            let node = self.force_load_single_object(file_index, is_level_0)?;
            nodes.push(node);
        }
        Ok(nodes)
    }

    // Retrieves an object from the cache, attempting to batch load if possible, based on the state of the batch load lock.
    //
    // This function first attempts to acquire the `batch_load_lock` using a non-blocking `try_lock`. If successful,
    // it sets a high `max_loads` value (1000), allowing for a larger batch load. This is the preferred scenario where
    // the system is capable of performing a more efficient batch load, loading multiple nodes at once. If the lock is
    // already held (i.e., another thread is performing a large batch load), the function falls back to a lower `max_loads`
    // value (1), effectively loading nodes one at a time to avoid blocking or deadlocking.
    //
    // The key idea here is to **always attempt to load as many nodes as possible** (with `max_loads = 1000`) unless
    // another thread is already performing a large load, in which case the function resorts to a smaller load size.
    // This dynamic loading strategy balances efficient batch loading with the need to avoid blocking or deadlocks in high-concurrency situations.
    //
    // After determining the appropriate `max_loads`, the function proceeds by calling `get_lazy_object`, which handles
    // the actual loading process, and retrieves the lazy-loaded data.
    pub fn get_object<T: ProbSerialize + ProbCacheable>(
        &self,
        file_index: FileIndex,
        is_level_0: bool,
    ) -> Result<*mut ProbLazyItem<T>, BufIoError> {
        let (_lock, max_loads) = match self.batch_load_lock.try_lock() {
            Ok(lock) => (Some(lock), 1000),
            Err(TryLockError::Poisoned(poison_err)) => panic!("lock error: {}", poison_err),
            Err(TryLockError::WouldBlock) => (None, 1),
        };
        self.get_lazy_object(file_index, max_loads, &mut HashSet::new(), is_level_0)
    }

    pub fn combine_index(file_index: &FileIndex, is_level_0: bool) -> u64 {
        let level_bit = if is_level_0 { 1u64 << 63 } else { 0 };
        match file_index {
            FileIndex::Valid {
                offset, version_id, ..
            } => ((offset.0 as u64) << 32) | (**version_id as u64) | level_bit,
            FileIndex::Invalid => u64::MAX, // Use max u64 value for Invalid
        }
    }

    pub fn get_prop_key(
        FileOffset(file_offset): FileOffset,
        BytesToRead(length): BytesToRead,
    ) -> u64 {
        (file_offset as u64) << 32 | (length as u64)
    }

    pub fn load_item<T: ProbSerialize>(
        &self,
        file_index: FileIndex,
        is_level_0: bool,
    ) -> Result<T, BufIoError> {
        let mut skipm: HashSet<u64> = HashSet::new();

        if file_index == FileIndex::Invalid {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Cannot deserialize with an invalid FileIndex",
            )
            .into());
        };

        T::deserialize(
            &self.bufmans,
            &self.level_0_bufmans,
            file_index,
            self,
            1000,
            &mut skipm,
            is_level_0,
        )
    }
}
