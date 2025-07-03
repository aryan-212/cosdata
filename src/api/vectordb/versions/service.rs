use std::sync::Arc;

use super::dtos::{CurrentVersionResponse, VersionListResponse, VersionMetadata};
use super::error::VersionError;
use crate::{app_context::AppContext, models::common::WaCustomError};

pub(crate) async fn list_versions(
    ctx: Arc<AppContext>,
    collection_id: &str,
) -> Result<VersionListResponse, VersionError> {
    // Get the collection to access transaction status map
    let collection = ctx
        .ain_env
        .collections_map
        .get_collection(collection_id)
        .ok_or_else(|| VersionError::DatabaseError("Collection not found".to_string()))?;
    let mut versions = collection
        .vcs
        .get_versions()
        .map_err(|e| WaCustomError::DatabaseError(e.to_string()))?;
    let current_version = *collection.current_version.read();

    // Patch the last version if it is an open implicit (epoch) version
    if let Some(last) = versions.last_mut() {
        use crate::models::versioning::VersionSource;
        if let VersionSource::Implicit { .. } = last.source {
            if let Some((open_version, upsert_count)) = collection.current_implicit_transaction.read().get_open_epoch_upsert_count() {
                if open_version == last.version {
                    let upserts = upsert_count.load(std::sync::atomic::Ordering::Relaxed);
                    last.records_upserted = upserts;
                }
            }
        }
    }

    // Calculate cumulative vector counts for each version
    let mut cumulative_count = 0u64;
    let versions = versions
        .into_iter()
        .map(|meta| {
            // Add this version's records to the cumulative total
            cumulative_count += meta.records_upserted as u64;

            VersionMetadata {
                version_number: meta.version,
                vector_count: cumulative_count,
            }
        })
        .collect::<Vec<VersionMetadata>>();
    Ok(VersionListResponse {
        versions,
        current_version,
    })
}

pub(crate) async fn get_current_version(
    ctx: Arc<AppContext>,
    collection_id: &str,
) -> Result<CurrentVersionResponse, VersionError> {
    // Get the collection to access transaction status map
    let collection = ctx
        .ain_env
        .collections_map
        .get_collection(collection_id)
        .ok_or_else(|| VersionError::DatabaseError("Collection not found".to_string()))?;
    let version_number = *collection.current_version.read();

    // Get vector count from transaction status for the current version
    // This should be the cumulative total up to the current version
    let vector_count = if let Ok(indexing_status) = collection.indexing_status() {
        indexing_status
            .status_summary
            .total_records_indexed_completed
    } else {
        // Fallback: calculate cumulative count manually
        let mut cumulative_count = 0u64;
        let versions = collection.vcs.get_versions().unwrap_or_default();
        for version_meta in versions {
            if *version_meta.version > *version_number {
                break; // Only count versions up to current version
            }
            cumulative_count += version_meta.records_upserted as u64;
        }
        cumulative_count
    };

    Ok(CurrentVersionResponse {
        version_number,
        vector_count,
    })
}

#[allow(unused)]
pub(crate) async fn set_current_version(
    _ctx: Arc<AppContext>,
    _collection_id: &str,
    _version_hash: &str,
) -> Result<(), VersionError> {
    // let collection = ctx.ain_env.collections_map.get(collection_id)
    //     .ok_or(VersionError::CollectionNotFound)?;

    // let hash_value = u32::from_str_radix(version_hash, 16)
    //     .map_err(|_| VersionError::InvalidVersionHash)?;
    // let hash = Hash::from(hash_value);

    // update_current_version(&collection.lmdb, hash)
    //     .map_err(|e| VersionError::UpdateFailed(e.to_string()))?;

    // Ok(())
    todo!()
}
