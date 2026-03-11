pub mod error;
pub mod model;
pub mod provider;
pub mod providers;
pub mod registry;
pub mod sse;
pub mod types;

pub use error::{BitAiError, Result};
pub use model::{Model, ModelInfo};
pub use provider::Provider;
pub use types::*;

use futures::stream::BoxStream;

/// Get a model handle by provider name and model ID.
pub fn model(provider: &str, model_id: &str) -> Result<Model> {
    model::get_model(provider, model_id)
}

/// Non-streaming chat completion.
pub async fn complete(model: &Model, req: ChatRequest) -> Result<ChatResponse> {
    model.provider.complete(&req).await
}

/// Streaming chat completion.
pub async fn stream(model: &Model, req: ChatRequest) -> Result<BoxStream<'_, Result<StreamEvent>>> {
    model.provider.stream(&req).await
}

/// List available models for a provider.
pub async fn list_models(provider: &str) -> Result<Vec<ModelInfo>> {
    let p = model::get_provider(provider)?;
    p.list_models().await
}

/// Register a custom provider.
pub fn register_provider(provider: std::sync::Arc<dyn Provider>) {
    model::register_provider(provider);
}

/// Initialize bit-ai: auto-register providers from environment variables.
pub fn init() {
    registry::auto_register();
}
