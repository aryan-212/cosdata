use std::{
    cell::UnsafeCell,
    marker::PhantomData,
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        Arc, RwLock,
    },
};

use super::{
    atomic_array::AtomicArray,
    buffered_io::{BufIoError, BufferManagerFactory},
    common::TSHashTable,
    serializer::{PartitionedSerialize, SimpleSerialize},
    tf_idf_index::{UnsafeVersionedVec, UnsafeVersionedVecIter},
    types::FileOffset,
    utils::calculate_path,
    versioning::VersionHash,
};

pub trait TreeMapKey: std::hash::Hash + Eq {
    fn key(&self) -> u64;
}

pub struct TreeMap<K, V> {
    pub(crate) root: TreeMapNode<V>,
    bufmans: BufferManagerFactory<u8>,
    _marker: PhantomData<K>,
}

pub struct TreeMapVec<K, V> {
    pub(crate) root: TreeMapVecNode<V>,
    bufmans: BufferManagerFactory<u8>,
    _marker: PhantomData<K>,
}

pub struct TreeMapNode<T> {
    pub node_idx: u16,
    pub offset: RwLock<Option<FileOffset>>,
    pub quotients: QuotientsMap<T>,
    pub children: AtomicArray<Self, 8>,
    pub dirty: AtomicBool,
}

pub struct TreeMapVecNode<T> {
    pub node_idx: u16,
    pub offset: RwLock<Option<FileOffset>>,
    pub quotients: QuotientsMapVec<T>,
    pub children: AtomicArray<Self, 8>,
    pub dirty: AtomicBool,
}

pub struct QuotientsMap<T> {
    pub offset: RwLock<Option<FileOffset>>,
    pub map: TSHashTable<u64, Arc<Quotient<T>>>,
    pub len: AtomicU64,
    pub serialized_upto: AtomicUsize,
}

pub struct QuotientsMapVec<T> {
    pub offset: RwLock<Option<FileOffset>>,
    pub map: TSHashTable<u64, Arc<QuotientVec<T>>>,
    pub len: AtomicU64,
    pub serialized_upto: AtomicUsize,
}

pub struct Quotient<T> {
    pub sequence_idx: u64,
    pub value: UnsafeVersionedItem<T>,
}

pub struct QuotientVec<T> {
    pub sequence_idx: u64,
    pub value: UnsafeVersionedVec<T>,
}

pub struct UnsafeVersionedItem<T> {
    pub serialized_at: RwLock<Option<FileOffset>>,
    pub version: VersionHash,
    pub value: T,
    pub next: UnsafeCell<Option<Box<Self>>>,
}

#[cfg(test)]
impl<T: PartialEq> PartialEq for TreeMapNode<T> {
    fn eq(&self, other: &Self) -> bool {
        *self.offset.read().unwrap() == *other.offset.read().unwrap()
            && self.quotients == other.quotients
            && self.children == other.children
    }
}

#[cfg(test)]
impl<T: PartialEq> PartialEq for TreeMapVecNode<T> {
    fn eq(&self, other: &Self) -> bool {
        *self.offset.read().unwrap() == *other.offset.read().unwrap()
            && self.quotients == other.quotients
            && self.children == other.children
    }
}

#[cfg(test)]
impl<T: std::fmt::Debug> std::fmt::Debug for TreeMapNode<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TreeMapNode")
            .field("offset", &*self.offset.read().unwrap())
            .field("quotients", &self.quotients)
            .field("children", &self.children)
            .finish()
    }
}

#[cfg(test)]
impl<T: std::fmt::Debug> std::fmt::Debug for TreeMapVecNode<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TreeMapVecNode")
            .field("offset", &*self.offset.read().unwrap())
            .field("quotients", &self.quotients)
            .field("children", &self.children)
            .finish()
    }
}

#[cfg(test)]
impl<T: PartialEq> PartialEq for QuotientsMap<T> {
    fn eq(&self, other: &Self) -> bool {
        self.map == other.map
            && self.len.load(Ordering::Relaxed) == other.len.load(Ordering::Relaxed)
            && self.serialized_upto.load(Ordering::Relaxed)
                == other.serialized_upto.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
impl<T: PartialEq> PartialEq for QuotientsMapVec<T> {
    fn eq(&self, other: &Self) -> bool {
        self.map == other.map
            && self.len.load(Ordering::Relaxed) == other.len.load(Ordering::Relaxed)
            && self.serialized_upto.load(Ordering::Relaxed)
                == other.serialized_upto.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
impl<T: std::fmt::Debug> std::fmt::Debug for QuotientsMap<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QuotientMap")
            .field("map", &self.map)
            .field("len", &self.len.load(Ordering::Relaxed))
            .field(
                "serialized_upto",
                &self.serialized_upto.load(Ordering::Relaxed),
            )
            .finish()
    }
}

#[cfg(test)]
impl<T: std::fmt::Debug> std::fmt::Debug for QuotientsMapVec<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QuotientMapVec")
            .field("map", &self.map)
            .field("len", &self.len.load(Ordering::Relaxed))
            .field(
                "serialized_upto",
                &self.serialized_upto.load(Ordering::Relaxed),
            )
            .finish()
    }
}

#[cfg(test)]
impl<T: PartialEq> PartialEq for Quotient<T> {
    fn eq(&self, other: &Self) -> bool {
        self.sequence_idx == other.sequence_idx && self.value == other.value
    }
}

#[cfg(test)]
impl<T: PartialEq> PartialEq for QuotientVec<T> {
    fn eq(&self, other: &Self) -> bool {
        self.sequence_idx == other.sequence_idx && self.value == other.value
    }
}

#[cfg(test)]
impl<T: std::fmt::Debug> std::fmt::Debug for Quotient<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Quotient")
            .field("sequence_idx", &self.sequence_idx)
            .field("value", &self.value)
            .finish()
    }
}

#[cfg(test)]
impl<T: std::fmt::Debug> std::fmt::Debug for QuotientVec<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QuotientVec")
            .field("sequence_idx", &self.sequence_idx)
            .field("value", &self.value)
            .finish()
    }
}

#[cfg(test)]
impl<T: PartialEq> PartialEq for UnsafeVersionedItem<T> {
    fn eq(&self, other: &Self) -> bool {
        self.version == other.version
            && self.value == other.value
            && unsafe { *self.next.get() == *other.next.get() }
    }
}

#[cfg(test)]
impl<T: std::fmt::Debug> std::fmt::Debug for UnsafeVersionedItem<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UnsafeVersionedItem")
            .field("version", &self.version)
            .field("value", &self.value)
            .field("next", unsafe { &*self.next.get() })
            .finish()
    }
}

impl<T> UnsafeVersionedItem<T> {
    pub fn new(version: VersionHash, value: T) -> Self {
        Self {
            serialized_at: RwLock::new(None),
            version,
            value,
            next: UnsafeCell::new(None),
        }
    }

    pub fn insert(&self, version: VersionHash, value: T) {
        self.insert_inner(Box::new(Self::new(version, value)));
    }

    fn insert_inner(&self, version: Box<Self>) {
        if let Some(next) = unsafe { &*self.next.get() } {
            return next.insert_inner(version);
        }

        unsafe {
            *self.next.get() = Some(version);
        }
    }

    pub fn latest(&self) -> &T {
        if let Some(next) = unsafe { &*self.next.get() } {
            return next.latest();
        }

        &self.value
    }

    pub fn latest_item(&self) -> &Self {
        if let Some(next) = unsafe { &*self.next.get() } {
            return next.latest_item();
        }

        self
    }
}

impl<T> TreeMapNode<T> {
    pub fn new(node_idx: u16) -> Self {
        Self {
            node_idx,
            offset: RwLock::new(None),
            quotients: QuotientsMap::default(),
            children: AtomicArray::default(),
            dirty: AtomicBool::new(true),
        }
    }

    pub fn find_or_create_node(&self, path: &[u8]) -> &Self {
        let mut current = self;
        for &idx in path {
            let new_node_idx = current.node_idx + (1u16 << (idx * 2));
            let (child, _) = current.children.get_or_insert(idx as usize, || {
                Box::into_raw(Box::new(Self::new(new_node_idx)))
            });
            current = unsafe { &*child };
        }
        current
    }

    pub fn insert(&self, version: VersionHash, quotient: u64, value: T) {
        self.quotients.insert(version, quotient, value);
        self.dirty.store(true, Ordering::Release);
    }

    pub fn get_latest(&self, quotient: u64) -> Option<&T> {
        self.quotients.get_latest(quotient)
    }

    pub fn get_versioned(&self, quotient: u64) -> Option<&UnsafeVersionedItem<T>> {
        self.quotients.get_versioned(quotient)
    }
}

impl<T> TreeMapVecNode<T> {
    pub fn new(node_idx: u16) -> Self {
        Self {
            node_idx,
            offset: RwLock::new(None),
            quotients: QuotientsMapVec::default(),
            children: AtomicArray::default(),
            dirty: AtomicBool::new(true),
        }
    }

    pub fn find_or_create_node(&self, path: &[u8]) -> &Self {
        let mut current = self;
        for &idx in path {
            let new_node_idx = current.node_idx + (1u16 << (idx * 2));
            let (child, _) = current.children.get_or_insert(idx as usize, || {
                Box::into_raw(Box::new(Self::new(new_node_idx)))
            });
            current = unsafe { &*child };
        }
        current
    }

    pub fn push(&self, version: VersionHash, quotient: u64, value: T) {
        self.quotients.push(version, quotient, value);
        self.dirty.store(true, Ordering::Release);
    }

    pub fn get(&self, quotient: u64) -> Option<UnsafeVersionedVecIter<'_, T>> {
        self.quotients.get(quotient)
    }
}

impl<T> Default for QuotientsMap<T> {
    fn default() -> Self {
        Self {
            offset: RwLock::new(None),
            map: TSHashTable::new(16),
            len: AtomicU64::new(0),
            serialized_upto: AtomicUsize::new(0),
        }
    }
}

impl<T> Default for QuotientsMapVec<T> {
    fn default() -> Self {
        Self {
            offset: RwLock::new(None),
            map: TSHashTable::new(16),
            len: AtomicU64::new(0),
            serialized_upto: AtomicUsize::new(0),
        }
    }
}

impl<T> QuotientsMap<T> {
    pub fn insert(&self, version: VersionHash, quotient: u64, value: T) {
        self.map.modify_or_insert_with_value(
            quotient,
            value,
            |value, quotient| {
                quotient.value.insert(version, value);
            },
            |value| {
                Arc::new(Quotient {
                    sequence_idx: self.len.fetch_add(1, Ordering::Relaxed),
                    value: UnsafeVersionedItem::new(version, value),
                })
            },
        );
    }

    fn get_latest(&self, quotient: u64) -> Option<&T> {
        self.map.lookup(&quotient).map(|q| {
            // SAFETY: here we are changing the lifetime of the value by using
            // `std::mem::transmute`, which by definition is not safe, but given our use of
            // this value and how we destroy it, its actually safe in this context.
            unsafe { std::mem::transmute(q.value.latest()) }
        })
    }

    fn get_versioned(&self, quotient: u64) -> Option<&UnsafeVersionedItem<T>> {
        self.map.lookup(&quotient).map(|q| {
            // SAFETY: here we are changing the lifetime of the value by using
            // `std::mem::transmute`, which by definition is not safe, but given our use of
            // this value and how we destroy it, its actually safe in this context.
            unsafe { std::mem::transmute(&q.value) }
        })
    }
}

impl<T> QuotientsMapVec<T> {
    pub fn push(&self, version: VersionHash, quotient: u64, value: T) {
        self.map.modify_or_insert_with_value(
            quotient,
            value,
            |value, quotient| {
                quotient.value.push(version, value);
            },
            |value| {
                let vec = UnsafeVersionedVec::new(version);
                vec.push(version, value);
                Arc::new(QuotientVec {
                    sequence_idx: self.len.fetch_add(1, Ordering::Relaxed),
                    value: vec,
                })
            },
        );
    }

    fn get(&self, quotient: u64) -> Option<UnsafeVersionedVecIter<'_, T>> {
        self.map.lookup(&quotient).map(|q| q.value.iter())
    }
}

#[cfg(test)]
impl<K: TreeMapKey + Clone, V: PartialEq> PartialEq for TreeMap<K, V> {
    fn eq(&self, other: &Self) -> bool {
        self.root == other.root
    }
}

#[cfg(test)]
impl<K: TreeMapKey + Clone, V: PartialEq> PartialEq for TreeMapVec<K, V> {
    fn eq(&self, other: &Self) -> bool {
        self.root == other.root
    }
}

#[cfg(test)]
impl<K: std::fmt::Debug + TreeMapKey + Clone + Ord, V: std::fmt::Debug> std::fmt::Debug
    for TreeMap<K, V>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TreeMap").field("root", &self.root).finish()
    }
}

#[cfg(test)]
impl<K: std::fmt::Debug + TreeMapKey + Clone + Ord, V: std::fmt::Debug> std::fmt::Debug
    for TreeMapVec<K, V>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TreeMapVec")
            .field("root", &self.root)
            .finish()
    }
}

impl<K: TreeMapKey, V> TreeMap<K, V> {
    pub fn new(bufmans: BufferManagerFactory<u8>) -> Self {
        Self {
            root: TreeMapNode::new(0),
            bufmans,
            _marker: PhantomData,
        }
    }
}

impl<K: TreeMapKey, V> TreeMap<K, V> {
    pub fn insert(&self, version: VersionHash, key: &K, value: V) {
        let key = key.key();
        let node_pos = (key % 65536) as u32;
        let path = calculate_path(node_pos, 0);
        let node = self.root.find_or_create_node(&path);
        node.insert(version, key, value);
    }

    pub fn get_latest(&self, key: &K) -> Option<&V> {
        let key = key.key();
        let node_pos = (key % 65536) as u32;
        let path = calculate_path(node_pos, 0);
        let node = self.root.find_or_create_node(&path);
        node.get_latest(key)
    }

    pub fn get_versioned(&self, key: &K) -> Option<&UnsafeVersionedItem<V>> {
        let key = key.key();
        let node_pos = (key % 65536) as u32;
        let path = calculate_path(node_pos, 0);
        let node = self.root.find_or_create_node(&path);
        node.get_versioned(key)
    }
}

impl<K, V: SimpleSerialize> TreeMap<K, V> {
    pub fn serialize(&self, file_parts: u8) -> Result<(), BufIoError> {
        let bufman = self.bufmans.get(0)?;
        let cursor = bufman.open_cursor()?;
        
        // Reserve space for root offset
        bufman.update_u32_with_cursor(cursor, u32::MAX)?;
        
        // Serialize with version tracking
        let offset = self.root.serialize_with_versions(&self.bufmans, file_parts, 0, 0)?;
        
        // Update root offset
        bufman.seek_with_cursor(cursor, 0)?;
        bufman.update_u32_with_cursor(cursor, offset)?;
        bufman.close_cursor(cursor)?;
        
        // Ensure all changes are flushed
        self.bufmans.flush_all()?;
        Ok(())
    }

    fn serialize_with_versions(
        &self,
        bufmans: &BufferManagerFactory<u8>,
        file_parts: u8,
        file_idx: u8,
        cursor: u64,
    ) -> Result<u32, BufIoError> {
        let file_idx = (self.node_idx % file_parts as u16) as u8;
        let bufman = bufmans.get(file_idx)?;
        let cursor = bufman.open_cursor()?;

        // Early return with version chain intact if already serialized
        if let Ok(guard) = self.offset.read() {
            if let Some(offset) = *guard {
                if !self.dirty.load(Ordering::Relaxed) {
                    return Ok(offset.0);
                }
            }
        }

        // Pre-allocate buffer for efficiency
        let mut buf = Vec::with_capacity(34 + self.children.items.len() * 4);
        buf.extend(self.node_idx.to_le_bytes());

        // Serialize children with version preservation
        for child in &self.children.items {
            let opt_child = unsafe { child.load(Ordering::Relaxed).as_ref() };
            match opt_child {
                Some(child) => {
                    let offset = child.serialize_with_versions(bufmans, file_parts, file_idx, cursor)?;
                    buf.extend(offset.to_le_bytes());
                }
                None => buf.extend(u32::MAX.to_le_bytes()),
            }
        }

        // Serialize quotients map with version info
        let quotients_offset = self.quotients.serialize_with_version(&bufman, cursor)?;
        buf.extend(quotients_offset.to_le_bytes());

        // Write buffer with minimal lock contention
        let start = bufman.write_to_end_of_file(cursor, &buf)? as u32;
        
        // Update offset with proper synchronization
        if let Ok(mut guard) = self.offset.write() {
            *guard = Some(FileOffset(start));
        }
        
        self.dirty.store(false, Ordering::Release);
        bufman.close_cursor(cursor)?;
        
        Ok(start)
    }

    pub fn deserialize(
        bufmans: BufferManagerFactory<u8>,
        file_parts: u8,
    ) -> Result<Self, BufIoError> {
        let bufman = bufmans.get(0)?;
        let cursor = bufman.open_cursor()?;
        let offset = bufman.read_u32_with_cursor(cursor)?;
        bufman.close_cursor(cursor)?;
        Ok(Self {
            root: TreeMapNode::deserialize(&bufmans, file_parts, 0, FileOffset(offset))?,
            bufmans,
            _marker: PhantomData,
        })
    }
}

impl<K: TreeMapKey, V> TreeMapVec<K, V> {
    pub fn new(bufmans: BufferManagerFactory<u8>) -> Self {
        Self {
            root: TreeMapVecNode::new(0),
            bufmans,
            _marker: PhantomData,
        }
    }
}

impl<K: TreeMapKey, V> TreeMapVec<K, V> {
    pub fn push(&self, version: VersionHash, key: &K, value: V) {
        let key = key.key();
        let node_pos = (key % 65536) as u32;
        let path = calculate_path(node_pos, 0);
        let node = self.root.find_or_create_node(&path);
        node.push(version, key, value);
    }

    pub fn get<'a>(&'a self, key: &K) -> Option<UnsafeVersionedVecIter<'a, V>> {
        let key = key.key();
        let node_pos = (key % 65536) as u32;
        let path = calculate_path(node_pos, 0);
        let node = self.root.find_or_create_node(&path);
        node.get(key)
    }
}

impl<K, V: SimpleSerialize> TreeMapVec<K, V> {
    pub fn serialize(&self, file_parts: u8) -> Result<(), BufIoError> {
        let bufman = self.bufmans.get(0)?;
        let cursor = bufman.open_cursor()?;
        bufman.update_u32_with_cursor(cursor, u32::MAX)?;
        let offset = self.root.serialize(&self.bufmans, file_parts, 0, 0)?;
        bufman.seek_with_cursor(cursor, 0)?;
        bufman.update_u32_with_cursor(cursor, offset)?;
        bufman.close_cursor(cursor)?;
        self.bufmans.flush_all()?;
        Ok(())
    }

    pub fn deserialize(
        bufmans: BufferManagerFactory<u8>,
        file_parts: u8,
    ) -> Result<Self, BufIoError> {
        let bufman = bufmans.get(0)?;
        let cursor = bufman.open_cursor()?;
        let offset = bufman.read_u32_with_cursor(cursor)?;
        bufman.close_cursor(cursor)?;
        Ok(Self {
            root: TreeMapVecNode::deserialize(&bufmans, file_parts, 0, FileOffset(offset))?,
            bufmans,
            _marker: PhantomData,
        })
    }
}

impl TreeMapKey for u64 {
    fn key(&self) -> u64 {
        *self
    }
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;

    #[test]
    fn tests_basic_usage() {
        let tempdir = tempdir().unwrap();
        let bufmans = BufferManagerFactory::new(
            tempdir.as_ref().into(),
            |root, part| root.join(format!("{}.tree-map", part)),
            8192,
        );
        let map: TreeMap<u64, u64> = TreeMap::new(bufmans);
        map.insert(0.into(), &65536, 0);
        map.insert(0.into(), &0, 23);
        assert_eq!(map.get_latest(&0), Some(&23));
        map.insert(1.into(), &0, 29);
        assert_eq!(map.get_latest(&0), Some(&29));
        assert_eq!(
            map.get_versioned(&0),
            Some(&UnsafeVersionedItem {
                version: 0.into(),
                serialized_at: RwLock::new(None),
                value: 23,
                next: UnsafeCell::new(Some(Box::new(UnsafeVersionedItem::new(1.into(), 29))))
            })
        );
    }
}
