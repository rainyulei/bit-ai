use async_trait::async_trait;
use futures::stream::BoxStream;

use crate::error::Result;
use crate::model::ModelInfo;
use crate::types::{ChatRequest, ChatResponse, StreamEvent};

#[async_trait]
pub trait Provider: Send + Sync {
    async fn complete(&self, req: &ChatRequest) -> Result<ChatResponse>;

    async fn stream(&self, req: &ChatRequest) -> Result<BoxStream<'_, Result<StreamEvent>>>;

    async fn list_models(&self) -> Result<Vec<ModelInfo>>;

    fn name(&self) -> &str;
}
