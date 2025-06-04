use rayon::prelude::*;

use std::{hash::Hasher, sync::RwLock};

use siphasher::sip::SipHasher24;

use crate::{
    config_loader::Config,
    models::{
        collection::Collection,
        collection_transaction::BackgroundCollectionTransaction,
        common::WaCustomError,
        meta_persist::MetaStore,
        types::{DocumentId, InternalId, VectorId},
    },
};

pub(crate) mod hnsw;
pub(crate) mod inverted;
pub(crate) mod tf_idf;

pub type InternalSearchResult = (
    InternalId,
    Option<VectorId>,
    Option<DocumentId>,
    f32,
    Option<String>,
);

pub type SearchResult = (VectorId, Option<DocumentId>, f32, Option<String>);

pub trait IndexOps: Send + Sync {
    type IndexingInput: Send + Sync;
    type SearchInput: Send + Sync;
    type SearchOptions: Send + Sync;
    type Data: serde::Serialize + serde::de::DeserializeOwned;

    fn validate_embedding(&self, embedding: Self::IndexingInput) -> Result<(), WaCustomError>;

    fn run_upload(
        &self,
        collection: &Collection,
        embeddings: Vec<Self::IndexingInput>,
        transaction: &BackgroundCollectionTransaction,
        config: &Config,
    ) -> Result<(), WaCustomError> {
        let Some(embeddings) = self.sample_embeddings(&collection.meta_store, embeddings, config)? else {
            return Ok(());
        };

        self.index_embeddings(collection, embeddings, transaction, config)
    }

    fn index_embeddings(
        &self,
        collection: &Collection,
        embeddings: Vec<Self::IndexingInput>,
        transaction: &BackgroundCollectionTransaction,
        config: &Config,
    ) -> Result<(), WaCustomError>;

    fn force_index(
        &self,
        collection: &Collection,
        transaction: &BackgroundCollectionTransaction,
        config: &Config,
    ) -> Result<(), WaCustomError> {
        if !self.is_configured() {
            let mut embeddings_guard = self.embeddings_collected().write().unwrap();
            self.finalize_sampling(&collection.meta_store, config, &embeddings_guard)?;
            let embeddings = std::mem::take(&mut *embeddings_guard);
            self.index_embeddings(collection, embeddings, transaction, config)?;
        }
        Ok(())
    }

    fn sample_embeddings(
        &self,
        meta_store: &MetaStore,
        sample_embeddings: Vec<Self::IndexingInput>,
        config: &Config,
    ) -> Result<Option<Vec<Self::IndexingInput>>, WaCustomError> {
        if self.is_configured() {
            return Ok(Some(sample_embeddings));
        }

        let collected_count = self.increment_collected_count(sample_embeddings.len());
        let sample_threshold = self.sample_threshold();

        if collected_count < sample_threshold {
            for embedding in &sample_embeddings {
                self.sample_embedding(embedding);
            }

            let mut collected_embeddings = self.embeddings_collected().write().unwrap();
            collected_embeddings.extend(sample_embeddings);
            if collected_embeddings.len() < sample_threshold {
                return Ok(None);
            }

            self.finalize_sampling(meta_store, config, &collected_embeddings)?;

            Ok(Some(std::mem::take(&mut *collected_embeddings)))
        } else {
            while !self.is_configured() {
                drop(self.embeddings_collected().read().unwrap());
            }
            Ok(Some(sample_embeddings))
        }
    }

    fn finalize_sampling(
        &self,
        meta_store: &MetaStore,
        config: &Config,
        embeddings: &[Self::IndexingInput],
    ) -> Result<(), WaCustomError>;

    fn sample_embedding(&self, embedding: &Self::IndexingInput);

    fn embeddings_collected(&self) -> &RwLock<Vec<Self::IndexingInput>>;

    fn increment_collected_count(&self, count: usize) -> usize;

    fn sample_threshold(&self) -> usize;

    // is this index configured? true is the sampling is done
    fn is_configured(&self) -> bool;

    // save everything to disk
    fn flush(&self, collection: &Collection) -> Result<(), WaCustomError>;

    fn pre_commit_transaction(
        &self,
        collection: &Collection,
        transaction: &BackgroundCollectionTransaction,
        config: &Config,
    ) -> Result<(), WaCustomError> {
        self.force_index(collection, transaction, config)?;
        self.flush(collection)
    }

    fn get_key_for_name(name: &str) -> u64 {
        let mut hasher = SipHasher24::new();
        hasher.write(name.as_bytes());
        hasher.finish()
    }

    fn get_data(&self) -> Self::Data;

    fn search_internal(
        &self,
        collection: &Collection,
        query: Self::SearchInput,
        options: &Self::SearchOptions,
        config: &Config,
        return_raw_text: bool,
    ) -> Result<Vec<InternalSearchResult>, WaCustomError>;

    fn remap_search_results(
        &self,
        collection: &Collection,
        results: Vec<InternalSearchResult>,
        return_raw_text: bool,
    ) -> Result<Vec<SearchResult>, WaCustomError> {
        results
            .into_iter()
            .map(|(internal_id, id, document_id, score, text)| {
                Ok(if let Some(id) = id {
                    (id, document_id, score, text)
                } else {
                    let raw_emb = collection
                        .internal_to_external_map
                        .get_latest(&internal_id)
                        .ok_or_else(|| {
                            WaCustomError::NotFound("raw embedding not found".to_string())
                        })?
                        .clone();
                    (
                        raw_emb.id.clone(),
                        raw_emb.document_id.clone(),
                        score,
                        if return_raw_text {
                            raw_emb.text.clone()
                        } else {
                            None
                        },
                    )
                })
            })
            .collect()
    }

    fn search(
        &self,
        collection: &Collection,
        query: Self::SearchInput,
        options: &Self::SearchOptions,
        config: &Config,
        return_raw_text: bool,
    ) -> Result<Vec<SearchResult>, WaCustomError> {
        let results = self.search_internal(collection, query, options, config, return_raw_text)?;
        self.remap_search_results(collection, results, return_raw_text)
    }

    fn batch_search(
        &self,
        collection: &Collection,
        queries: Vec<Self::SearchInput>,
        options: &Self::SearchOptions,
        config: &Config,
        return_raw_text: bool,
    ) -> Result<Vec<Vec<SearchResult>>, WaCustomError> {
        queries
            .into_par_iter()
            .map(|query| self.search(collection, query, options, config, return_raw_text))
            .collect()
    }
}
