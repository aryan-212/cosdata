use std::sync::{
    atomic::{AtomicU16, AtomicU32, Ordering},
    Arc, RwLock,
};

use super::{TFIDFIndexSerialize, TF_IDF_INDEX_DATA_CHUNK_SIZE};
use crate::models::{
    buffered_io::{BufIoError, BufferManager, BufferManagerFactory},
    cache_loader::TFIDFIndexCache,
    common::TSHashTable,
    tf_idf_index::{TFIDFIndexNodeData, TermInfo},
    types::FileOffset,
};

impl TFIDFIndexNodeData {
    pub fn batch_insert(
        &self,
        version: VersionHash,
        entries: Vec<(TermQuotient, (f32, u32))>,
    ) -> Result<(), BufIoError> {
        // Sort entries by quotient for efficient insertion
        let mut sorted_entries = entries;
        sorted_entries.sort_by_key(|(q, _)| *q);

        // Group by quotient for batch updates
        let mut current_quotient = None;
        let mut current_batch = Vec::new();

        for (quotient, value) in sorted_entries {
            match current_quotient {
                Some(q) if q == quotient => {
                    current_batch.push(value);
                }
                _ => {
                    // Process previous batch if any
                    if let Some(q) = current_quotient {
                        if let Some(term_info) = self.terms.get(&q) {
                            term_info.batch_push(version, &current_batch);
                        }
                    }
                    // Start new batch
                    current_quotient = Some(quotient);
                    current_batch.clear();
                    current_batch.push(value);
                }
            }
        }

        // Process final batch
        if let Some(quotient) = current_quotient {
            if let Some(term_info) = self.terms.get(&quotient) {
                term_info.batch_push(version, &current_batch);
            }
        }

        Ok(())
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
        // Start with header
        let mut header_buf = Vec::with_capacity(16);
        header_buf.extend(TF_IDF_INDEX_DATA_CHUNK_SIZE.to_le_bytes());
        
        let start = offset_counter.fetch_add(header_buf.len() as u32, Ordering::AcqRel);
        dim_bufman.write_to_end_of_file(cursor, &header_buf)?;

        // Sort terms for deterministic serialization
        let mut term_entries: Vec<_> = self.terms.iter().collect();
        term_entries.sort_by_key(|(k, _)| *k);

        // Batch serialize terms with optimized buffer usage
        for chunk in term_entries.chunks(TF_IDF_INDEX_DATA_CHUNK_SIZE as usize) {
            let mut chunk_buf = Vec::with_capacity(chunk.len() * 16);
            
            for (quotient, term_info) in chunk {
                chunk_buf.extend(quotient.to_le_bytes());
                let term_offset = term_info.serialize(
                    dim_bufman,
                    data_bufmans, 
                    offset_counter,
                    file_idx,
                    file_parts,
                    cursor,
                )?;
                chunk_buf.extend(term_offset.to_le_bytes());
            }

            dim_bufman.write_to_end_of_file(cursor, &chunk_buf)?;
        }

        Ok(start)
    }
}

impl TFIDFIndexSerialize for TFIDFIndexNodeData {
    fn serialize(
        &self,
        dim_bufman: &BufferManager,
        data_bufmans: &BufferManagerFactory<u8>,
        offset_counter: &AtomicU32,
        data_file_idx: u8,
        data_file_parts: u8,
        cursor: u64,
    ) -> Result<u32, BufIoError> {
        let start = dim_bufman.cursor_position(cursor)? as u32;
        let mut list = self.map.to_list();
        let mut num_entries_serialized = self.num_entries_serialized.write().unwrap();
        let map_len = self.map_len.load(Ordering::Acquire);
        debug_assert_eq!(map_len, list.len() as u16);
        list.sort_unstable_by_key(|(_, v)| v.sequence_idx);
        dim_bufman.update_u16_with_cursor(cursor, map_len)?;

        let total_chunks = list.len().div_ceil(TF_IDF_INDEX_DATA_CHUNK_SIZE);

        for chunk_idx in 0..total_chunks {
            for i in (chunk_idx * TF_IDF_INDEX_DATA_CHUNK_SIZE)
                ..((chunk_idx + 1) * TF_IDF_INDEX_DATA_CHUNK_SIZE)
            {
                let current_offset = dim_bufman.cursor_position(cursor)?;
                if *num_entries_serialized > i as u16 {
                    list[i].1.serialize(
                        dim_bufman,
                        data_bufmans,
                        offset_counter,
                        data_file_idx,
                        data_file_parts,
                        cursor,
                    )?;
                    dim_bufman.seek_with_cursor(cursor, current_offset + 6)?;
                } else if let Some((quotient, term)) = list.get(i) {
                    let offset = term.serialize(
                        dim_bufman,
                        data_bufmans,
                        offset_counter,
                        data_file_idx,
                        data_file_parts,
                        cursor,
                    )?;
                    dim_bufman.seek_with_cursor(cursor, current_offset)?;
                    dim_bufman.update_u16_with_cursor(cursor, *quotient)?;
                    dim_bufman.update_u32_with_cursor(cursor, offset)?;
                } else {
                    dim_bufman.update_with_cursor(cursor, &[u8::MAX; 6])?;
                }
            }
            if *num_entries_serialized > ((chunk_idx + 1) * TF_IDF_INDEX_DATA_CHUNK_SIZE) as u16 {
                let offset = dim_bufman.read_u32_with_cursor(cursor)?;
                dim_bufman.seek_with_cursor(cursor, offset as u64)?;
            } else if chunk_idx == total_chunks - 1 {
                dim_bufman.update_with_cursor(cursor, &[u8::MAX; 4])?;
            } else {
                let offset = offset_counter.fetch_add(
                    (6 * TF_IDF_INDEX_DATA_CHUNK_SIZE + 4) as u32,
                    Ordering::Relaxed,
                );
                dim_bufman.update_u32_with_cursor(cursor, offset)?;
                dim_bufman.seek_with_cursor(cursor, offset as u64)?;
            }
        }

        *num_entries_serialized = list.len() as u16;

        Ok(start)
    }

    fn deserialize(
        dim_bufman: &BufferManager,
        data_bufmans: &BufferManagerFactory<u8>,
        file_offset: FileOffset,
        data_file_idx: u8,
        data_file_parts: u8,
        cache: &TFIDFIndexCache,
    ) -> Result<Self, BufIoError> {
        let cursor = dim_bufman.open_cursor()?;
        dim_bufman.seek_with_cursor(cursor, file_offset.0 as u64)?;
        let map_len = dim_bufman.read_u16_with_cursor(cursor)? as usize;
        let map = TSHashTable::new(16);

        let total_chunks = map_len.div_ceil(TF_IDF_INDEX_DATA_CHUNK_SIZE);

        for chunk_idx in 0..total_chunks {
            for i in (chunk_idx * TF_IDF_INDEX_DATA_CHUNK_SIZE)
                ..((chunk_idx + 1) * TF_IDF_INDEX_DATA_CHUNK_SIZE)
            {
                if i == map_len {
                    break;
                }
                let quotient = dim_bufman.read_u16_with_cursor(cursor)?;
                let term_offset = dim_bufman.read_u32_with_cursor(cursor)?;
                let mut term = TermInfo::deserialize(
                    dim_bufman,
                    data_bufmans,
                    FileOffset(term_offset),
                    data_file_idx,
                    data_file_parts,
                    cache,
                )?;
                term.sequence_idx = i as u16;
                map.insert(quotient, Arc::new(term));
            }

            let next_chunk_offset = dim_bufman.read_u32_with_cursor(cursor)?;
            if next_chunk_offset == u32::MAX {
                break;
            }
            dim_bufman.seek_with_cursor(cursor, next_chunk_offset as u64)?;
        }

        Ok(Self {
            map,
            map_len: AtomicU16::new(map_len as u16),
            num_entries_serialized: RwLock::new(map_len as u16),
        })
    }
}
