use std::sync::Arc;
use std::path::Path;

use crate::{
    app_context::AppContext,
    models::{
        collection::{Collection, CollectionIndexingStatus},
        common::WaCustomError,
        tree_map::TreeMap,
    },
};

use super::{
    dtos::{
        CollectionSummaryDto, CreateCollectionDto, GetCollectionsDto, GetCollectionsResponseDto,
        ListCollectionsResponseDto,
    },
    error::CollectionsError,
};

use crate::models::buffered_io::BufferManagerFactory;
use crate::models::versioning::VersionNumber;

pub(crate) async fn create_collection(
    ctx: Arc<AppContext>,
    CreateCollectionDto {
        name,
        description,
        config,
        dense_vector,
        metadata_schema,
        sparse_vector,
        tf_idf_options,
        store_raw_text,
    }: CreateCollectionDto,
) -> Result<Arc<Collection>, CollectionsError> {
    // Check if collection already exists
    if ctx.ain_env.collections_map.get_collection(&name).is_some() {
        return Err(CollectionsError::AlreadyExists(name));
    }

    // Construct the BufferManagerFactory for MetaStore
    let collections_path: Arc<Path> = crate::models::paths::get_data_path().join("collections").join(&name).into();
    let meta_store_bufmans = BufferManagerFactory::new(
        collections_path.clone(),
        |root, part| root.join(format!("{}.meta", part)),
        8192,
    );
    let meta_store = Arc::new(TreeMap::new(meta_store_bufmans));

    // VersionControl and hash logic may need to be updated to not use LMDB
    // For now, pass dummy values or refactor as needed
    let vcs = panic!("VersionControl construction for Treemap-based collections is not yet implemented");
    let hash = VersionNumber::from(0); // Placeholder, update as needed

    let metadata_schema = match metadata_schema {
        Some(s) => {
            let schema = s
                .try_into()
                .map_err(|e| CollectionsError::WaCustomError(WaCustomError::MetadataError(e)))?;
            Some(schema)
        }
        None => None,
    };

    let collection = Collection::new(
        name,
        description,
        dense_vector,
        sparse_vector,
        tf_idf_options,
        metadata_schema,
        config,
        store_raw_text,
        meta_store,
        hash,
        vcs,
        &ctx,
    )
    .map_err(CollectionsError::WaCustomError)?;

    // adding the created collection into the in-memory map
    ctx.ain_env
        .collections_map
        .insert_collection(collection.clone())
        .map_err(CollectionsError::WaCustomError)?;

    // Remove LMDB-specific persistence and update_current_version
    collection
        .flush(&ctx.config)
        .map_err(CollectionsError::WaCustomError)?;
    // update_current_version(&collection.meta_store, hash).map_err(CollectionsError::WaCustomError)?;
    Ok(collection)
}

/// gets a list of collections
/// TODO results should be filtered based on search params,
/// if no params provided, it returns all collections
pub(crate) async fn get_collections(
    ctx: Arc<AppContext>,
    _get_collections_dto: GetCollectionsDto,
) -> Result<Vec<GetCollectionsResponseDto>, CollectionsError> {
    let collections = ctx
        .ain_env
        .collections_map
        .iter_collections()
        .map(|c| GetCollectionsResponseDto {
            name: c.meta.name.clone(),
            description: c.meta.description.clone(),
        })
        .collect();
    Ok(collections)
}

/// gets a collection by its name
pub(crate) async fn get_collection_by_name(
    ctx: Arc<AppContext>,
    name: &str,
) -> Result<Arc<Collection>, CollectionsError> {
    let collection = match ctx.ain_env.collections_map.get_collection(name) {
        Some(collection) => collection.clone(),
        None => {
            // dense index not found, return an error response
            return Err(CollectionsError::NotFound);
        }
    };
    Ok(collection)
}

pub(crate) async fn get_collection_indexing_status(
    ctx: Arc<AppContext>,
    name: &str,
) -> Result<CollectionIndexingStatus, CollectionsError> {
    let collection = match ctx.ain_env.collections_map.get_collection(name) {
        Some(collection) => collection.clone(),
        None => {
            // dense index not found, return an error response
            return Err(CollectionsError::NotFound);
        }
    };
    collection
        .indexing_status()
        .map_err(CollectionsError::WaCustomError)
}

pub(crate) async fn delete_collection_by_name(
    ctx: Arc<AppContext>,
    name: &str,
) -> Result<Arc<Collection>, CollectionsError> {
    let collection = get_collection_by_name(ctx.clone(), name).await?;

    // Remove collection from in-memory map
    let collection = ctx
        .ain_env
        .collections_map
        .remove_collection(name)
        .map_err(CollectionsError::WaCustomError)?;

    Ok(collection)
}

pub(crate) async fn list_collections(
    ctx: Arc<AppContext>,
) -> Result<ListCollectionsResponseDto, CollectionsError> {
    // Iterate over collections stored in the AppContext map
    let summaries: Vec<CollectionSummaryDto> = ctx
        .ain_env
        .collections_map
        .iter_collections()
        .map(|collection_arc| CollectionSummaryDto {
            name: collection_arc.meta.name.clone(),
            description: collection_arc.meta.description.clone(),
        })
        .collect();

    Ok(ListCollectionsResponseDto {
        collections: summaries,
    })
}
