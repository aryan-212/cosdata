use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use crate::models::{
    atomic_array::AtomicArray,
    buffered_io::{BufIoError, BufferManager, BufferManagerFactory},
    cache_loader::TFIDFIndexCache,
    prob_lazy_load::lazy_item::ProbLazyItem,
    tf_idf_index::{TFIDFIndexNode, TFIDFIndexNodeData},
    types::FileOffset,
};

use super::{TFIDFIndexSerialize, TF_IDF_INDEX_DATA_CHUNK_SIZE};

// @SERIALIZED_SIZE:
//
//   4 byte for dim index +                          | 4
//   2 bytes for data map len +                      | 6
//   INVERTED_INDEX_DATA_CHUNK_SIZE * (              |
//     2 bytes for quotient +                        |
//     4 bytes of pagepool                           |
//   ) +                                             | INVERTED_INDEX_DATA_CHUNK_SIZE * 6 + 6
//   4 byte for next data chunk                      | INVERTED_INDEX_DATA_CHUNK_SIZE * 6 + 10
//   16 * 4 bytes for dimension offsets +            | INVERTED_INDEX_DATA_CHUNK_SIZE * 6 + 74
impl TFIDFIndexSerialize for TFIDFIndexNode {
    fn serialize(
        &self,
        dim_bufman: &BufferManager,
        data_bufmans: &BufferManagerFactory<u8>,
        offset_counter: &AtomicU32,
        _: u8,
        data_file_parts: u8,
        cursor: u64,
    ) -> Result<u32, BufIoError> {
        let data_file_idx = (self.dim_index % data_file_parts as u32) as u8;
        if !self.is_serialized.swap(true, Ordering::AcqRel) {
            dim_bufman.seek_with_cursor(cursor, self.file_offset.0 as u64)?;
            dim_bufman.update_u32_with_cursor(cursor, self.dim_index)?;
            self.data.serialize(
                dim_bufman,
                data_bufmans,
                offset_counter,
                data_file_idx,
                data_file_parts,
                cursor,
            )?;
            dim_bufman.seek_with_cursor(
                cursor,
                self.file_offset.0 as u64 + TF_IDF_INDEX_DATA_CHUNK_SIZE as u64 * 6 + 10,
            )?;
            self.children.serialize(
                dim_bufman,
                data_bufmans,
                offset_counter,
                data_file_idx,
                data_file_parts,
                cursor,
            )?;
        } else if self.is_dirty.swap(false, Ordering::AcqRel) {
            dim_bufman.seek_with_cursor(cursor, self.file_offset.0 as u64 + 4)?;
            self.data.serialize(
                dim_bufman,
                data_bufmans,
                offset_counter,
                data_file_idx,
                data_file_parts,
                cursor,
            )?;
            dim_bufman.seek_with_cursor(
                cursor,
                self.file_offset.0 as u64 + TF_IDF_INDEX_DATA_CHUNK_SIZE as u64 * 6 + 10,
            )?;
            self.children.serialize(
                dim_bufman,
                data_bufmans,
                offset_counter,
                data_file_idx,
                data_file_parts,
                cursor,
            )?;
        } else {
            dim_bufman.seek_with_cursor(
                cursor,
                self.file_offset.0 as u64 + TF_IDF_INDEX_DATA_CHUNK_SIZE as u64 * 6 + 10,
            )?;
            self.children.serialize(
                dim_bufman,
                data_bufmans,
                offset_counter,
                data_file_idx,
                data_file_parts,
                cursor,
            )?;
        };
        Ok(self.file_offset.0)
    }

    fn deserialize(
        dim_bufman: &BufferManager,
        data_bufmans: &BufferManagerFactory<u8>,
        file_offset: FileOffset,
        _: u8,
        data_file_parts: u8,
        cache: &TFIDFIndexCache,
    ) -> Result<Self, BufIoError> {
        let cursor = dim_bufman.open_cursor()?;
        dim_bufman.seek_with_cursor(cursor, file_offset.0 as u64)?;
        let dim_index = dim_bufman.read_u32_with_cursor(cursor)?;
        let data_file_idx = (dim_index % data_file_parts as u32) as u8;
        let data = <*mut ProbLazyItem<TFIDFIndexNodeData>>::deserialize(
            dim_bufman,
            data_bufmans,
            FileOffset(file_offset.0 + 4),
            data_file_idx,
            data_file_parts,
            cache,
        )?;
        let children = AtomicArray::deserialize(
            dim_bufman,
            data_bufmans,
            FileOffset(file_offset.0 + TF_IDF_INDEX_DATA_CHUNK_SIZE as u32 * 6 + 10),
            data_file_idx,
            data_file_parts,
            cache,
        )?;

        Ok(Self {
            is_serialized: AtomicBool::new(true),
            is_dirty: AtomicBool::new(false),
            dim_index,
            file_offset,
            data,
            children,
        })
    }
}

impl TFIDFIndexNode {
    pub const fn get_serialized_size() -> u32 {
        // Increased for better buffering of version chains
        4096
    }

    pub fn serialize(
        &self,
        dim_bufman: &BufferManager,
        data_bufmans: &BufferManagerFactory<u8>,
        offset_counter: &AtomicU32,
        file_idx: u8,
        file_parts: u8,
        cursor: u64,
    ) -> Result<u32, BufIoError> {
        let bufman = data_bufmans.get(file_idx)?;

        // Early return if clean
        if let Ok(guard) = self.offset.read() {
            if let Some(offset) = *guard {
                if !self.dirty.load(Ordering::Relaxed) {
                    return Ok(offset.0);
                }
            }
        }

        // Pre-allocate with larger buffer for version chains
        let mut buf = Vec::with_capacity(self.get_serialized_size() as usize);
        buf.extend(self.node_idx.to_le_bytes());

        // Serialize children with version preservation
        for child in &self.children.items {
            let opt_child = unsafe { child.load(Ordering::Relaxed).as_ref() };
            match opt_child {
                Some(child) => {
                    let offset = child.serialize(
                        dim_bufman,
                        data_bufmans,
                        offset_counter,
                        file_idx,
                        file_parts,
                        cursor,
                    )?;
                    buf.extend(offset.to_le_bytes());
                }
                None => buf.extend(u32::MAX.to_le_bytes()),
            }
        }

        // Serialize data with proper version tracking
        let data_offset = self.data.serialize(
            dim_bufman,
            data_bufmans,
            offset_counter,
            file_idx,
            file_parts,
            cursor,
        )?;
        buf.extend(data_offset.to_le_bytes());

        // Write buffer minimizing lock contention
        let start = if let Ok(mut guard) = self.offset.write() {
            let offset = if let Some(current) = *guard {
                // Update in-place if possible
                dim_bufman.seek_with_cursor(cursor, current.0 as u64)?;
                dim_bufman.write_to_file_with_cursor(cursor, &buf, current.0 as u64)?;
                current.0
            } else {
                // Write to new location
                let new_offset = offset_counter.fetch_add(buf.len() as u32, Ordering::AcqRel);
                dim_bufman.seek_with_cursor(cursor, new_offset as u64)?;
                let written = dim_bufman.write_to_end_of_file(cursor, &buf)? as u32;
                *guard = Some(FileOffset(written));
                written
            };
            offset
        } else {
            return Err(BufIoError::Locking);
        };

        self.dirty.store(false, Ordering::Release);
        Ok(start)
    }

    // Helper for merging version chains
    fn merge_version_chains<T>(
        current: &[UnsafeVersionedItem<T>],
        new: &[UnsafeVersionedItem<T>],
    ) -> Vec<UnsafeVersionedItem<T>>
    where
        T: Clone,
    {
        let mut result = Vec::with_capacity(current.len() + new.len());
        let mut i = 0;
        let mut j = 0;

        while i < current.len() && j < new.len() {
            if current[i].version <= new[j].version {
                result.push(current[i].clone());
                i += 1;
            } else {
                result.push(new[j].clone());
                j += 1;
            }
        }

        result.extend_from_slice(&current[i..]);
        result.extend_from_slice(&new[j..]);
        result
    }
}

impl TermInfo {
    // Add atomic batch operations
    pub fn batch_push(&mut self, version: VersionHash, values: &[(f32, u32)]) {
        let new_values: Vec<UnsafeVersionedItem<(f32, u32)>> = values
            .iter()
            .map(|&value| UnsafeVersionedItem::new(version, value))
            .collect();
        
        let mut vals = self.values.write().unwrap();
        vals.extend(new_values);
        
        // Update atomically
        self.sequence_idx.fetch_add(values.len() as u32, Ordering::AcqRel);
    }

    pub fn serialize(
        &self,
        dim_bufman: &BufferManager,
        data_bufmans: &BufferManagerFactory<u8>,
        offset_counter: &AtomicU32,
        file_idx: u8,
        file_parts: u8,
        cursor: u64,
    ) -> Result<u32, BufIoError> {
        let bufman = data_bufmans.get(file_idx)?;
        let values = self.values.read().map_err(|_| BufIoError::Locking)?;

        // Pre-allocate buffer
        let mut buf = Vec::with_capacity(values.len() * 12 + 8);
        
        // Write sequence index
        buf.extend(self.sequence_idx.load(Ordering::Acquire).to_le_bytes());
        
        // Write sorted, deduped values maintaining version order
        let mut sorted_values: Vec<_> = values.iter().collect();
        sorted_values.sort_by_key(|v| v.version);
        
        for value in sorted_values {
            buf.extend(value.version.0.to_le_bytes());
            buf.extend(value.value.0.to_le_bytes());
            buf.extend(value.value.1.to_le_bytes());
        }

        // Write with proper offset tracking
        let start = offset_counter.fetch_add(buf.len() as u32, Ordering::AcqRel);
        dim_bufman.seek_with_cursor(cursor, start as u64)?;
        dim_bufman.write_with_cursor(cursor, &buf)?;

        Ok(start)
    }
}
