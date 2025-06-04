use crate::models::common::*;
use crate::models::versioning::*;
use crate::models::tree_map::{TreeMap, TreeMapKey};
use serde::{Serialize, Deserialize};
use std::sync::Arc;
use bincode;

/// MetaStore type using TreeMap for storing metadata
pub type MetaStore = Arc<TreeMap<String, Vec<u8>>>;

/// updates the current version of a collection
pub fn update_current_version(
    store: &MetaStore,
    version_hash: VersionNumber,
) -> Result<(), WaCustomError> {
    let key = "current_version".to_string();
    let bytes = version_hash.to_le_bytes().to_vec();
    store.as_ref().insert(0.into(), &key, bytes);
    Ok(())
}

/// updates the background version of a collection
pub fn update_background_version(
    store: &MetaStore,
    version: VersionNumber,
) -> Result<(), WaCustomError> {
    let key = "background_version".to_string();
    let bytes = version.to_le_bytes().to_vec();
    store.as_ref().insert(0.into(), &key, bytes);
    Ok(())
}

pub fn store_values_range(store: &MetaStore, range: (f32, f32)) -> Result<(), WaCustomError> {
    let key = "values_range".to_string();
    let mut bytes = Vec::with_capacity(8);
    bytes.extend(range.0.to_le_bytes());
    bytes.extend(range.1.to_le_bytes());
    store.as_ref().insert(0.into(), &key, bytes);
    Ok(())
}

pub fn store_values_upper_bound(store: &MetaStore, bound: f32) -> Result<(), WaCustomError> {
    let key = "values_upper_bound".to_string();
    let bytes = bound.to_le_bytes().to_vec();
    store.as_ref().insert(0.into(), &key, bytes);
    Ok(())
}

pub fn store_average_document_length(store: &MetaStore, len: f32) -> Result<(), WaCustomError> {
    let key = "average_document_length".to_string();
    let bytes = len.to_le_bytes().to_vec();
    store.as_ref().insert(0.into(), &key, bytes);
    Ok(())
}

pub fn store_highest_internal_id(store: &MetaStore, id: u32) -> Result<(), WaCustomError> {
    let key = "highest_internal_id".to_string();
    let bytes = id.to_le_bytes().to_vec();
    store.as_ref().insert(0.into(), &key, bytes);
    Ok(())
}

/// retrieves the current version of a collection
pub fn retrieve_current_version(store: &MetaStore) -> Result<VersionNumber, WaCustomError> {
    let key = "current_version".to_string();
    let bytes = store.as_ref().get_latest(&key).ok_or_else(|| WaCustomError::DatabaseError("Record not found: current_version".to_string()))?;
    let bytes: [u8; 4] = bytes.as_slice().try_into().map_err(|_| WaCustomError::DeserializationError("Failed to deserialize Hash: length mismatch".to_string()))?;
    Ok(VersionNumber::from(u32::from_le_bytes(bytes)))
}

/// retrieves the background version of a collection
pub fn retrieve_background_version(store: &MetaStore) -> Result<VersionNumber, WaCustomError> {
    let key = "background_version".to_string();
    let bytes = store.as_ref().get_latest(&key).ok_or_else(|| WaCustomError::DatabaseError("Record not found: background_version".to_string()))?;
    let bytes: [u8; 4] = bytes.as_slice().try_into().map_err(|_| WaCustomError::DeserializationError("Failed to deserialize Hash: length mismatch".to_string()))?;
    Ok(VersionNumber::from(u32::from_le_bytes(bytes)))
}

pub fn retrieve_values_range(store: &MetaStore) -> Result<Option<(f32, f32)>, WaCustomError> {
    let key = "values_range".to_string();
    let bytes = match store.as_ref().get_latest(&key) {
        Some(bytes) => bytes,
        None => return Ok(None),
    };
    let bytes: [u8; 8] = bytes.as_slice().try_into().map_err(|_| WaCustomError::DeserializationError("Failed to deserialize values range: length mismatch".to_string()))?;
    let start = f32::from_le_bytes(bytes[..4].try_into().unwrap());
    let end = f32::from_le_bytes(bytes[4..].try_into().unwrap());
    Ok(Some((start, end)))
}

pub fn retrieve_values_upper_bound(store: &MetaStore) -> Result<Option<f32>, WaCustomError> {
    let key = "values_upper_bound".to_string();
    let bytes = match store.as_ref().get_latest(&key) {
        Some(bytes) => bytes,
        None => return Ok(None),
    };
    let bytes: [u8; 4] = bytes.as_slice().try_into().map_err(|_| WaCustomError::DeserializationError("Failed to deserialize values upper bound: length mismatch".to_string()))?;
    Ok(Some(f32::from_le_bytes(bytes)))
}

pub fn retrieve_average_document_length(store: &MetaStore) -> Result<Option<f32>, WaCustomError> {
    let key = "average_document_length".to_string();
    let bytes = match store.as_ref().get_latest(&key) {
        Some(bytes) => bytes,
        None => return Ok(None),
    };
    let bytes: [u8; 4] = bytes.as_slice().try_into().map_err(|_| WaCustomError::DeserializationError("Failed to deserialize average document length: length mismatch".to_string()))?;
    Ok(Some(f32::from_le_bytes(bytes)))
}

pub fn retrieve_highest_internal_id(store: &MetaStore) -> Result<Option<u32>, WaCustomError> {
    let key = "highest_internal_id".to_string();
    let bytes = match store.as_ref().get_latest(&key) {
        Some(bytes) => bytes,
        None => return Ok(None),
    };
    let bytes: [u8; 4] = bytes.as_slice().try_into().map_err(|_| WaCustomError::DeserializationError("Failed to deserialize highest internal id: length mismatch".to_string()))?;
    Ok(Some(u32::from_le_bytes(bytes)))
}

// The following functions for initializing/loading collections are now obsolete and should be removed or refactored to use Treemap if needed.
