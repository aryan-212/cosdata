mod controller;
mod delete;
mod dtos;
mod error;
mod repo;
mod service;
mod update;

use actix_web::{web, Scope};
pub(crate) use delete::delete;
pub(crate) use update::update;

pub(crate) fn transactions_module() -> Scope {
    web::scope("/collections/{collection_id}/transactions")
        .route("", web::post().to(controller::create_transaction))
        .route(
            "/{transaction_id}/commit",
            web::post().to(controller::commit_transaction),
        )
        .route(
            "/{transaction_id}/status",
            web::get().to(controller::get_transaction_status),
        )
        .route(
            "/{transaction_id}/vectors",
            web::post().to(controller::create_vector_in_transaction),
        )
        .route(
            "/{transaction_id}/upsert",
            web::post().to(controller::upsert),
        )
        .route(
            "/{transaction_id}/vectors/{vector_id}",
            web::delete().to(controller::delete_vector_by_id),
        )
        .route(
            "/{transaction_id}/abort",
            web::post().to(controller::abort_transaction),
        )
}
