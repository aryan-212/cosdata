use super::{
    buffered_io::{BufIoError, BufferManagerFactory},
    cache_loader::HNSWIndexCache,
    collection::{Collection, CollectionMetadata, DenseVectorOptions, SparseVectorOptions, TFIDFOptions, CollectionConfig},
    crypto::{DoubleSHA256Hash, SingleSHA256Hash},
    indexing_manager::IndexingManager,
    inverted_index::InvertedIndexRoot,
    meta_persist::{
        retrieve_average_document_length, retrieve_background_version, retrieve_current_version,
        retrieve_highest_internal_id, retrieve_values_upper_bound,
    },
    paths::get_data_path,
    prob_node::ProbNode,
    tf_idf_index::TFIDFIndexRoot,
    tree_map::{TreeMap, TreeMapKey, TreeMapVec},
    versioning::{VersionControl, VersionNumber},
};
use crate::{
    args::CosdataArgs,
    config_loader::{Config, Server, Host, Port, Ssl, ServerMode, Hnsw, Indexing, VectorsIndexingMode, Search, CacheConfig},
    distance::{
        cosine::{CosineDistance, CosineSimilarity},
        dotproduct::DotProductDistance,
        euclidean::EuclideanDistance,
        hamming::HammingDistance,
        DistanceError, DistanceFunction,
    },
    indexes::{
        hnsw::{
            offset_counter::{HNSWIndexFileOffsetCounter, IndexFileId},
            HNSWIndex,
        },
        inverted::InvertedIndex,
        tf_idf::TFIDFIndex,
        IndexOps,
    },
    metadata::{schema::MetadataDimensions, QueryFilterDimensions, HIGH_WEIGHT},
    models::{
        buffered_io::BufferManager,
        common::*,
        lazy_item::{FileIndex, ProbLazyItem},
        meta_persist::retrieve_values_range,
        prob_node::{LatestNode, SharedLatestNode},
        serializer::hnsw::RawDeserialize,
    },
    quantization::{
        product::ProductQuantization, scalar::ScalarQuantization, Quantization, QuantizationError,
        StorageType,
    },
    storage::Storage,
    app_context::AppContext,
};
use crossbeam::channel;
use dashmap::DashMap;
use lmdb::{Cursor, Database, DatabaseFlags, Environment, Transaction, WriteFlags};
use rayon::ThreadPool;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use siphasher::sip::SipHasher24;
use std::{
    fmt,
    fs::{self, create_dir_all, OpenOptions},
    hash::{Hash as StdHash, Hasher},
    io::Write,
    ops::{Deref, Div, Mul},
    path::{Path, PathBuf},
    str::FromStr,
    sync::{
        atomic::{AtomicBool, AtomicU32, AtomicUsize},
        Arc, RwLock,
    },
    thread,
    time::Instant,
};
use super::meta_persist::MetaStore;
use bincode;
use log::warn;
use super::collection_cache::CollectionCacheManager;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct HNSWLevel(pub u8);

#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct FileOffset(pub u32);

#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct BytesToRead(pub u32);

pub type PropPersistRef = (FileOffset, BytesToRead);

#[derive(Debug, PartialEq)]
pub struct NodePropValue {
    pub id: InternalId,
    pub vec: Arc<Storage>,
    pub location: PropPersistRef,
}

impl StdHash for NodePropValue {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.id.hash(state);
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Metadata {
    pub mag: f32,
    pub mbits: Vec<i32>,
}

impl From<MetadataDimensions> for Metadata {
    fn from(dims: MetadataDimensions) -> Self {
        let total = dims
            .iter()
            .map(|d| {
                let x = *d as f32;
                x * x
            })
            .sum::<f32>();
        // @NOTE: As `MetadataDimensions` have high weight values, we
        // need to handle overflow during intermediate addition when
        // calculating the euclidean norm
        let mag = total.min(f32::MAX).sqrt();
        Self { mag, mbits: dims }
    }
}

impl From<&QueryFilterDimensions> for Metadata {
    fn from(dims: &QueryFilterDimensions) -> Self {
        let dims_i32 = dims.iter().map(|d| *d as i32).collect::<Vec<i32>>();
        // @NOTE: Unlike `MetadataDimensions`, `QueryFilterDimensions`
        // will have -1, 0, 1 values so no need to worry about
        // overflow during summation
        let mag = dims
            .iter()
            .map(|d| {
                let x = *d as f32;
                x * x
            })
            .sum::<f32>()
            .sqrt();
        Self {
            mag,
            mbits: dims_i32,
        }
    }
}

impl PartialEq for Metadata {
    fn eq(&self, other: &Self) -> bool {
        self.mag.to_bits() == other.mag.to_bits() && self.mbits == other.mbits
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataId(pub u8);

#[derive(Debug, PartialEq)]
pub struct NodePropMetadata {
    pub vec: Arc<Metadata>,
    pub location: PropPersistRef,
}

#[derive(Debug)]
pub enum ReplicaNodeKind {
    Pseudo,
    Base,
    Metadata,
}

#[derive(Debug)]
pub struct VectorData<'a> {
    // Vector id (use specified one and not the internal replica
    // id). It's not being used any where but occasionally useful for
    // debugging. In case it's a query vector, `id` expected to be
    // None.
    pub id: Option<&'a InternalId>,
    pub quantized_vec: &'a Storage,
    pub metadata: Option<&'a Metadata>,
}

impl<'a> VectorData<'a> {
    pub fn without_metadata(id: Option<&'a InternalId>, qvec: &'a Storage) -> Self {
        Self {
            id,
            quantized_vec: qvec,
            metadata: None,
        }
    }

    pub fn replica_node_kind(&self) -> ReplicaNodeKind {
        match self.metadata {
            Some(m) => {
                if m.mag == 0.0 {
                    ReplicaNodeKind::Base
                } else {
                    match self.id {
                        Some(id) => {
                            if ((u32::MAX - 257)..=(u32::MAX - 2)).contains(&**id) {
                                ReplicaNodeKind::Pseudo
                            } else {
                                ReplicaNodeKind::Metadata
                            }
                        }
                        None => ReplicaNodeKind::Metadata,
                    }
                }
            }
            None => ReplicaNodeKind::Base,
        }
    }

    pub fn is_pseudo_root(&self) -> bool {
        match self.metadata {
            Some(m) => m.mbits == vec![HIGH_WEIGHT; m.mbits.len()],
            None => false,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Hash, Eq, PartialEq)]
pub struct VectorId(String);

impl From<String> for VectorId {
    fn from(id: String) -> Self {
        Self(id)
    }
}

impl Deref for VectorId {
    type Target = String;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<VectorId> for String {
    fn from(id: VectorId) -> Self {
        id.0
    }
}

impl TreeMapKey for VectorId {
    fn key(&self) -> u64 {
        let mut hasher = SipHasher24::new();
        self.0.hash(&mut hasher);
        hasher.finish()
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Hash, PartialEq, Eq)]
pub struct DocumentId(String);

impl From<String> for DocumentId {
    fn from(id: String) -> Self {
        Self(id)
    }
}

impl Deref for DocumentId {
    type Target = String;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<DocumentId> for String {
    fn from(id: DocumentId) -> Self {
        id.0
    }
}

impl TreeMapKey for DocumentId {
    fn key(&self) -> u64 {
        let mut hasher = SipHasher24::new();
        self.0.hash(&mut hasher);
        hasher.finish()
    }
}

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
    Serialize,
    Deserialize,
)]
pub struct InternalId(u32);

impl From<u32> for InternalId {
    fn from(id: u32) -> Self {
        Self(id)
    }
}

impl Deref for InternalId {
    type Target = u32;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<InternalId> for u32 {
    fn from(id: InternalId) -> Self {
        id.0
    }
}

impl TreeMapKey for InternalId {
    fn key(&self) -> u64 {
        self.0 as u64
    }
}

impl Div<u32> for InternalId {
    type Output = Self;

    fn div(self, rhs: u32) -> Self::Output {
        Self(self.0 / rhs)
    }
}

impl Mul<u32> for InternalId {
    type Output = Self;

    fn mul(self, rhs: u32) -> Self::Output {
        Self(self.0 * rhs)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub enum MetricResult {
    CosineSimilarity(CosineSimilarity),
    // @DOUBT: how can we obtain `CosineDistance`?
    CosineDistance(CosineDistance),
    EuclideanDistance(EuclideanDistance),
    HammingDistance(HammingDistance),
    // @DOUBT: dot product shows similarity between two vectors, not distance,
    // should rename it to `DotProduct`?
    DotProductDistance(DotProductDistance),
}

impl PartialOrd for MetricResult {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for MetricResult {}

impl Ord for MetricResult {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self {
            Self::CosineSimilarity(val) => val.0.total_cmp(&other.get_value()),
            Self::CosineDistance(val) => other.get_value().total_cmp(&val.0),
            Self::EuclideanDistance(val) => other.get_value().total_cmp(&val.0),
            Self::HammingDistance(val) => other.get_value().total_cmp(&val.0),
            Self::DotProductDistance(val) => val.0.total_cmp(&other.get_value()),
        }
    }
}

impl MetricResult {
    // gets the bare numerical value stored in the type
    pub fn get_value(&self) -> f32 {
        match self {
            MetricResult::CosineSimilarity(value) => value.0,
            MetricResult::CosineDistance(value) => value.0,
            MetricResult::EuclideanDistance(value) => value.0,
            MetricResult::HammingDistance(value) => value.0,
            MetricResult::DotProductDistance(value) => value.0,
        }
    }

    pub fn get_tag_and_value(&self) -> (u8, f32) {
        match self {
            Self::CosineSimilarity(value) => (0, value.0),
            Self::CosineDistance(value) => (1, value.0),
            Self::EuclideanDistance(value) => (2, value.0),
            Self::HammingDistance(value) => (3, value.0),
            Self::DotProductDistance(value) => (4, value.0),
        }
    }

    pub fn min(metric: DistanceMetric) -> Self {
        match metric {
            DistanceMetric::Cosine => Self::CosineSimilarity(CosineSimilarity(-1.0)),
            DistanceMetric::Euclidean => {
                Self::EuclideanDistance(EuclideanDistance(f32::NEG_INFINITY))
            }
            DistanceMetric::Hamming => Self::HammingDistance(HammingDistance(f32::NEG_INFINITY)),
            DistanceMetric::DotProduct => {
                Self::DotProductDistance(DotProductDistance(f32::NEG_INFINITY))
            }
        }
    }

    pub fn max(metric: DistanceMetric) -> Self {
        match metric {
            DistanceMetric::Cosine => Self::CosineSimilarity(CosineSimilarity(2.0)), // take care of precision issues
            DistanceMetric::Euclidean => Self::EuclideanDistance(EuclideanDistance(f32::INFINITY)),
            DistanceMetric::Hamming => Self::HammingDistance(HammingDistance(f32::INFINITY)),
            DistanceMetric::DotProduct => {
                Self::DotProductDistance(DotProductDistance(f32::INFINITY))
            }
        }
    }
}

#[derive(Debug, Clone, Copy, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DistanceMetric {
    Cosine,
    Euclidean,
    Hamming,
    DotProduct,
}

impl DistanceFunction for DistanceMetric {
    type Item = MetricResult;
    fn calculate(
        &self,
        x: &VectorData,
        y: &VectorData,
        is_indexing: bool,
    ) -> Result<Self::Item, DistanceError> {
        match self {
            Self::Cosine => {
                let value = CosineSimilarity(0.0).calculate(x, y, is_indexing)?;
                Ok(MetricResult::CosineSimilarity(value))
            }
            Self::Euclidean => {
                let value = EuclideanDistance(0.0).calculate(x, y, is_indexing)?;
                Ok(MetricResult::EuclideanDistance(value))
            }
            Self::Hamming => {
                let value = HammingDistance(0.0).calculate(x, y, is_indexing)?;
                Ok(MetricResult::HammingDistance(value))
            }
            Self::DotProduct => {
                let value = DotProductDistance(0.0).calculate(x, y, is_indexing)?;
                Ok(MetricResult::DotProductDistance(value))
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantizationMetric {
    Scalar,
    Product(ProductQuantization),
}

impl Quantization for QuantizationMetric {
    fn quantize(
        &self,
        vector: &[f32],
        storage_type: StorageType,
        range: (f32, f32),
    ) -> Result<Storage, QuantizationError> {
        match self {
            Self::Scalar => ScalarQuantization.quantize(vector, storage_type, range),
            Self::Product(product) => product.quantize(vector, storage_type, range),
        }
    }

    fn train(&mut self, vectors: &[&[f32]]) -> Result<(), QuantizationError> {
        match self {
            Self::Scalar => ScalarQuantization.train(vectors),
            Self::Product(product) => product.train(vectors),
        }
    }
}

// Implementing the std::fmt::Display trait for VectorId
impl fmt::Display for VectorId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone)]
pub struct MetaDb {
    pub env: Arc<Environment>,
    pub db: Database,
}

impl MetaDb {
    pub fn from_env(env: Arc<Environment>, collection_name: &str) -> lmdb::Result<Self> {
        let db = env.create_db(Some(collection_name), DatabaseFlags::empty())?;

        Ok(Self { env, db })
    }
}

pub struct CollectionsMap {
    inner_collections: DashMap<String, Arc<Collection>>,
}

impl CollectionsMap {
    pub fn new() -> Self {
        Self {
            inner_collections: DashMap::new(),
        }
    }

    /// Loads collections map from the collections directory using Treemap-based MetaStore
    pub fn load_from_disk(config: Arc<Config>, threadpool: Arc<ThreadPool>) -> Result<Self, WaCustomError> {
        let collections_path = get_collections_path();
        let mut map = Self::new();
        if collections_path.exists() {
            for entry in fs::read_dir(&collections_path).map_err(|e| WaCustomError::FsError(e.to_string()))? {
                let entry = entry.map_err(|e| WaCustomError::FsError(e.to_string()))?;
                let path = entry.path();
                if path.is_dir() {
                    let name = path.file_name().unwrap().to_string_lossy().to_string();
                    let meta_store_bufmans = BufferManagerFactory::new(
                        path.clone().into(),
                        |root, part| root.join(format!("{}.meta", part)),
                8192,
            );
                    let meta_store = Arc::new(TreeMap::<String, Vec<u8>>::new(meta_store_bufmans));
                    // Load CollectionMetadata from meta_store
                    let meta_bytes = match meta_store.as_ref().get_latest(&"collection_metadata".to_string()) {
                        Some(bytes) => bytes,
                        None => {
                            warn!("No metadata found for collection '{}', skipping", name);
                            continue;
                        }
                    };
                    let meta: CollectionMetadata = match bincode::deserialize(meta_bytes) {
                        Ok(m) => m,
            Err(e) => {
                            warn!("Failed to deserialize metadata for collection '{}': {}", name, e);
                            continue;
                        }
                    };
                    // Load current_version from meta_store
                    let current_version = match retrieve_current_version(&meta_store) {
                        Ok(v) => v,
            Err(e) => {
                            warn!("Failed to load current_version for collection '{}': {}", name, e);
                            continue;
                        }
                    };
                    // Use Option for vcs and ctx to allow compilation
                    let dummy_env = Arc::new(Environment::new().set_max_dbs(1).open(std::path::Path::new("/tmp")).unwrap());
                    let dummy_db = dummy_env.create_db(Some("dummy"), DatabaseFlags::empty()).unwrap();
                    let vcs = VersionControl::from_existing(dummy_env, dummy_db);
                    let dummy_config = Arc::new(Config {
                        thread_pool: crate::config_loader::ThreadPool { pool_size: 1 },
                        server: Server {
                            host: Host::Hostname("localhost".to_string()),
                            port: Port::from(8080),
                            ssl: Ssl {
                                cert_file: std::path::PathBuf::from("dummy.crt"),
                                key_file: std::path::PathBuf::from("dummy.key"),
                            },
                            mode: ServerMode::Http,
                        },
                        hnsw: Hnsw {
                            default_neighbors_count: 1,
                            default_level_0_neighbors_count: 1,
                            default_ef_construction: 1,
                            default_ef_search: 1,
                            default_num_layer: 1,
                            default_max_cache_size: 1,
                        },
                        indexing: Indexing {
                            clamp_margin_percent: 0.0,
                            mode: VectorsIndexingMode::Sequential,
                        },
                        search: Search {
                            shortlist_size: 1,
                            early_terminate_threshold: 0.0,
                        },
                        upload_threshold: 1,
                        upload_process_batch_size: 1,
                        num_regions_to_load_on_restart: 1,
                        inverted_index_data_file_parts: 1,
                        tree_map_serialized_parts: 1,
                        sparse_raw_values_reranking_factor: 1,
                        rerank_sparse_with_raw_values: false,
                        index_file_min_size: 1,
                        cache: CacheConfig::default(),
                    });
                    let dummy_threadpool = Arc::new(rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap());
                    let dummy_app_env = Arc::new(AppEnv {
                        collections_map: CollectionsMap::new(),
                        users_map: UsersMap::new(Arc::new(Environment::new().set_max_dbs(1).open(std::path::Path::new("/tmp")).unwrap())).unwrap(),
                        persist: Arc::new(Environment::new().set_max_dbs(1).open(std::path::Path::new("/tmp")).unwrap()),
                        admin_key: SingleSHA256Hash([0u8; 32]),
                        active_sessions: Arc::new(DashMap::new()),
                    });
                    let ctx = AppContext {
                        config: dummy_config.clone(),
                        threadpool: dummy_threadpool.clone(),
                        ain_env: dummy_app_env.clone(),
                        collection_cache_manager: Arc::new(CollectionCacheManager::new(
                            Arc::from(std::path::Path::new("/tmp")),
                            1, // max_collections
                            0.1, // eviction_probability
                            dummy_app_env.clone(),
                        )),
                    };
                    let collection = Collection::new(
                        name.clone(),
                        meta.description.clone(),
                        meta.dense_vector.clone(),
                        meta.sparse_vector.clone(),
                        meta.tf_idf_options.clone(),
                        meta.metadata_schema.clone(),
                        meta.config.clone(),
                        meta.store_raw_text,
                        meta_store,
                        current_version,
                        vcs,
                        &ctx,
                    )?;
                    map.inner_collections.insert(name, collection);
                }
            }
        }
        Ok(map)
    }

    pub fn insert_collection(&self, collection: Arc<Collection>) -> Result<(), WaCustomError> {
        self.inner_collections.insert(collection.meta.name.clone(), collection);
        Ok(())
    }

    pub fn get_collection(&self, name: &str) -> Option<Arc<Collection>> {
        self.inner_collections.get(name).map(|c| c.clone())
    }

    pub fn remove_collection(&self, name: &str) -> Result<Arc<Collection>, WaCustomError> {
        self.inner_collections.remove(name).map(|(_, c)| c).ok_or(WaCustomError::NotFound("collection not found".to_string()))
    }

    pub fn iter_collections(&self) -> dashmap::iter::Iter<String, Arc<Collection>, std::hash::RandomState, DashMap<String, Arc<Collection>>> {
        self.inner_collections.iter()
    }
}

pub struct UsersMap {
    env: Arc<Environment>,
    users_db: Database,
    // (username, user details)
    map: DashMap<String, User>,
}

impl UsersMap {
    pub fn new(env: Arc<Environment>) -> lmdb::Result<Self> {
        let users_db = env.create_db(Some("users"), DatabaseFlags::empty())?;
        let txn = env.begin_ro_txn()?;
        let mut cursor = txn.open_ro_cursor(users_db)?;
        let map = DashMap::new();

        for (username, user_bytes) in cursor.iter() {
            let username = String::from_utf8(username.to_vec()).unwrap();
            let user = User::deserialize(user_bytes).unwrap();
            map.insert(username, user);
        }

        drop(cursor);
        txn.abort();

        Ok(Self { env, users_db, map })
    }

    pub fn add_user(&self, username: String, password_hash: DoubleSHA256Hash) -> lmdb::Result<()> {
        let user = User {
            username: username.clone(),
            password_hash,
        };
        let user_bytes = user.serialize();
        let username_bytes = username.as_bytes();

        let mut txn = self.env.begin_rw_txn()?;
        txn.put(
            self.users_db,
            &username_bytes,
            &user_bytes,
            WriteFlags::empty(),
        )?;
        txn.commit()?;

        self.map.insert(username, user);

        Ok(())
    }

    pub fn get_user(&self, username: &str) -> Option<User> {
        self.map.get(username).map(|user| user.value().clone())
    }
}

#[derive(Clone)]
pub struct User {
    pub username: String,
    pub password_hash: DoubleSHA256Hash,
}

impl User {
    fn serialize(&self) -> Vec<u8> {
        let username_bytes = self.username.as_bytes();
        let mut buf = Vec::with_capacity(32 + username_bytes.len());
        buf.extend_from_slice(&self.password_hash.0);
        buf.extend_from_slice(username_bytes);
        buf
    }

    fn deserialize(buf: &[u8]) -> Result<Self, String> {
        if buf.len() < 32 {
            return Err("Input must be at least 32 bytes".to_string());
        }
        let mut password_hash = [0u8; 32];
        password_hash.copy_from_slice(&buf[..32]);
        let username_bytes = buf[32..].to_vec();
        let username = String::from_utf8(username_bytes).map_err(|err| err.to_string())?;
        Ok(Self {
            username,
            password_hash: DoubleSHA256Hash(password_hash),
        })
    }
}

pub struct SessionDetails {
    pub created_at: u64,
    pub expires_at: u64,
    pub user: User,
}

// Define the AppEnv struct
pub struct AppEnv {
    pub collections_map: CollectionsMap,
    pub users_map: UsersMap,
    pub persist: Arc<Environment>,
    // Single hash, must not be persisted to disk, only the double hash must be
    // written to disk
    pub admin_key: SingleSHA256Hash,
    pub active_sessions: Arc<DashMap<String, SessionDetails>>,
}

fn get_admin_key(env: Arc<Environment>, args: CosdataArgs) -> lmdb::Result<SingleSHA256Hash> {
    // Create meta database if it doesn't exist
    let init_txn = env.begin_rw_txn()?;
    unsafe { init_txn.create_db(Some("meta"), DatabaseFlags::empty())? };
    init_txn.commit()?;

    let txn = env.begin_ro_txn()?;
    let db = unsafe { txn.open_db(Some("meta"))? };

    let admin_key_from_lmdb = match txn.get(db, &"admin_key") {
        Ok(bytes) => {
            let mut hash_array = [0u8; 32];
            // Copy bytes from the database to the fixed-size array
            if bytes.len() >= 32 {
                hash_array.copy_from_slice(&bytes[..32]);
                Some(DoubleSHA256Hash(hash_array))
            } else {
                log::error!("Invalid admin key format in database");
                return Err(lmdb::Error::Other(7));
            }
        }
        Err(lmdb::Error::NotFound) => None,
        Err(e) => return Err(e),
    };
    txn.abort();

    let admin_key_hash = if let Some(admin_key_from_lmdb) = admin_key_from_lmdb {
        // Database already exists, verify admin key
        let arg_admin_key = args.admin_key;
        let arg_admin_key_hash = SingleSHA256Hash::from_str(&arg_admin_key).unwrap();
        let arg_admin_key_double_hash = arg_admin_key_hash.hash_again();
        if !admin_key_from_lmdb.verify_eq(&arg_admin_key_double_hash) {
            log::error!("Invalid admin key!");
            return Err(lmdb::Error::Other(5));
        }
        arg_admin_key_hash
    } else {
        // First-time setup
        let arg_admin_key = args.admin_key;
        let arg_admin_key_hash = SingleSHA256Hash::from_str(&arg_admin_key).unwrap();
        let arg_admin_key_double_hash = arg_admin_key_hash.hash_again();

        // Store the admin key double hash in the database
        let mut txn = env.begin_rw_txn()?;
        let db = unsafe { txn.open_db(Some("meta"))? };
        txn.put(
            db,
            &"admin_key",
            &arg_admin_key_double_hash.0,
            WriteFlags::empty(),
        )?;
        txn.commit()?;
        arg_admin_key_hash
    };
    Ok(admin_key_hash)
}

pub fn get_collections_path() -> PathBuf {
    get_data_path().join("collections")
}

pub fn get_app_env(
    config: Arc<Config>,
    threadpool: Arc<ThreadPool>,
    args: CosdataArgs,
) -> Result<Arc<AppEnv>, WaCustomError> {
    // Check both possible db path locations
    let db_path_1 = get_data_path().join("_mdb");
    let db_path_2 = get_data_path().join("data/_mdb");

    // Use whichever path exists, or default to db_path_2
    let db_path = if db_path_1.exists() {
        //println!("Using existing database at {}", db_path_1.display());
        db_path_1
    } else if db_path_2.exists() {
        //println!("Using existing database at {}", db_path_2.display());
        db_path_2
    } else {
        //println!("Creating new database at {}", db_path_2.display());
        db_path_2 // Default for first-time setup
    };

    // Check if this is first-time setup
    let is_first_time = !db_path.exists();

    // If this is first time and confirmation is required
    if is_first_time && !args.skip_confirmation && !args.confirmed {
        // Interactive prompt for confirmation
        print!("Re-enter admin key: ");
        std::io::stdout().flush().unwrap();

        let mut confirmation = String::new();
        std::io::stdin().read_line(&mut confirmation).map_err(|e| {
            WaCustomError::ConfigError(format!("Failed to read confirmation: {}", e))
        })?;

        // Remove trailing newline
        confirmation = confirmation.trim().to_string();

        if confirmation != args.admin_key {
            return Err(WaCustomError::ConfigError(
                "Admin key and confirmation do not match".to_string(),
            ));
        }

        println!("Admin key confirmed successfully.");
    }

    // Create a modified args with confirmed flag set
    let mut confirmed_args = args.clone();
    confirmed_args.confirmed = true;

    // Ensure parent directories exist first if needed
    if let Some(parent) = db_path.parent() {
        create_dir_all(parent).map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;
    }

    // Ensure the database directory exists
    create_dir_all(&db_path).map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;
    // Initialize the environment
    let env = Environment::new()
        .set_max_dbs(10)
        .set_map_size(1048576000) // Set the maximum size of the database to 1GB
        .open(&db_path)
        .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;

    let env_arc = Arc::new(env);

    let admin_key = get_admin_key(env_arc.clone(), confirmed_args)
        .map_err(|err| WaCustomError::DatabaseError(err.to_string()))?;

    // Add more resilient error handling for collections_map loading
    let collections_map = match CollectionsMap::load_from_disk(config, threadpool) {
        Ok(map) => map,
        Err(_e) => {
            //println!("Warning: Failed to load collections map: {}", e);
            //println!("Creating a new collections map...");
            // Use the correct function signature for CollectionsMap::new
            CollectionsMap::new()
        }
    };

    let users_map = match UsersMap::new(env_arc.clone()) {
        Ok(map) => map,
        Err(err) => {
            println!("Warning: Failed to load users map: {}", err);
            return Err(WaCustomError::DatabaseError(err.to_string()));
        }
    };

    // Use the admin key as the password instead of hardcoded "admin"
    let username = "admin".to_string();
    let password = args.admin_key.clone();
    let password_hash = DoubleSHA256Hash::from_str(&password).unwrap();

    // Don't fail if user already exists
    match users_map.add_user(username, password_hash) {
        Ok(_) => {}
        Err(err) => {
            println!(
                "Note: Could not add admin user (may already exist): {}",
                err
            );
        }
    };

    Ok(Arc::new(AppEnv {
        collections_map,
        users_map,
        persist: env_arc,
        admin_key,
        active_sessions: Arc::new(DashMap::new()),
    }))
}

#[allow(unused)]
#[derive(Debug, Clone)]
pub struct SparseVector {
    pub vector_id: u32,
    pub entries: Vec<(u32, f32)>,
}

impl SparseVector {
    #[allow(unused)]
    pub fn new(vector_id: u32, entries: Vec<(u32, f32)>) -> Self {
        Self { vector_id, entries }
    }
}

#[cfg(test)]
mod tests {
    use crate::distance::cosine::CosineSimilarity;

    use super::MetricResult;

    #[test]
    fn test_metric_result_ordering() {
        let mut metric_results = vec![
            MetricResult::CosineSimilarity(CosineSimilarity(6.0)),
            MetricResult::CosineSimilarity(CosineSimilarity(5.0)),
            MetricResult::CosineSimilarity(CosineSimilarity(4.0)),
            MetricResult::CosineSimilarity(CosineSimilarity(3.0)),
            MetricResult::CosineSimilarity(CosineSimilarity(2.0)),
            MetricResult::CosineSimilarity(CosineSimilarity(1.0)),
        ];

        let correctly_ordered_metric_results = vec![
            MetricResult::CosineSimilarity(CosineSimilarity(1.0)),
            MetricResult::CosineSimilarity(CosineSimilarity(2.0)),
            MetricResult::CosineSimilarity(CosineSimilarity(3.0)),
            MetricResult::CosineSimilarity(CosineSimilarity(4.0)),
            MetricResult::CosineSimilarity(CosineSimilarity(5.0)),
            MetricResult::CosineSimilarity(CosineSimilarity(6.0)),
        ];

        metric_results.sort();

        assert_eq!(metric_results, correctly_ordered_metric_results);
    }
}
