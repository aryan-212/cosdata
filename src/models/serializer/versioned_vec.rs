use std::{cell::UnsafeCell, sync::RwLock};

use crate::models::{
    buffered_io::{BufIoError, BufferManager},
    tf_idf_index::UnsafeVersionedVec,
    types::FileOffset,
    versioning::VersionHash,
};

use super::SimpleSerialize;

impl<T: SimpleSerialize> SimpleSerialize for UnsafeVersionedVec<T> {
    fn serialize(&self, bufman: &BufferManager, cursor: u64) -> Result<u32, BufIoError> {
        // Serialize next node first
        let next = unsafe { &*self.next.get() };
        let next_offset = if let Some(next) = next {
            next.serialize(bufman, cursor)?
        } else {
            u32::MAX
        };

        // Early return if already serialized
        if let Ok(guard) = self.serialized_at.read() {
            if let Some(offset) = *guard {
                // Update next pointer and return
                bufman.seek_with_cursor(cursor, offset.0 as u64)?;
                bufman.update_u32_with_cursor(cursor, next_offset)?;
                return Ok(offset.0);
            }
        }

        // Prepare serialization buffer
        let list = unsafe { &*self.list.get() };
        let mut buf = Vec::with_capacity(8 + list.len() * 4); // Pre-allocate buffer

        // Write version
        buf.extend(self.version.to_le_bytes());

        // Write list length and contents
        buf.extend((list.len() as u32).to_le_bytes());
        for item in list {
            let serialized_offset = item.serialize(bufman, cursor)?;
            buf.extend(serialized_offset.to_le_bytes());
        }

        // Write to file
        let written_bytes = bufman.write_to_end_of_file(cursor, &buf)?;
        let offset: u32 = written_bytes.try_into().map_err(|_| {
            BufIoError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "File offset too large"
            ))
        })?;

        // Update serialized_at with minimal lock time
        if let Ok(mut guard) = self.serialized_at.write() {
            *guard = Some(FileOffset(offset));
        }

        Ok(offset)
    }

    fn deserialize(bufman: &BufferManager, offset: FileOffset) -> Result<Self, BufIoError> {
        let cursor = bufman.open_cursor()?;
        bufman.seek_with_cursor(cursor, offset.0 as u64)?;

        // Read version
        let version = VersionHash::from(bufman.read_u64_with_cursor(cursor)?);

        // Read list length and contents
        let len = bufman.read_u32_with_cursor(cursor)? as usize;
        let mut list = Vec::with_capacity(len);
        for _ in 0..len {
            let el_offset = bufman.read_u32_with_cursor(cursor)?;
            let el = T::deserialize(bufman, FileOffset(el_offset))?;
            list.push(el);
        }

        // Create UnsafeVersionedVec with loaded data
        let next = None; // Next pointer will be handled elsewhere if needed
        bufman.close_cursor(cursor)?;
        
        Ok(Self {
            serialized_at: RwLock::new(Some(offset)),
            version,
            list: UnsafeCell::new(list),
            next: UnsafeCell::new(next),
        })
    }
}
