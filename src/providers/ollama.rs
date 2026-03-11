use async_trait::async_trait;
use futures::stream::BoxStream;

use crate::error::Result;
use crate::model::ModelInfo;
use crate::provider::Provider;
use crate::providers::openai_compat::{OpenAICompatConfig, OpenAICompatProvider};
use crate::types::*;

pub struct OllamaProvider {
    inner: OpenAICompatProvider,
}

impl OllamaProvider {
    pub fn new() -> Self {
        Self {
            inner: OpenAICompatProvider::new(OpenAICompatConfig {
                name: "ollama".to_string(),
                base_url: "http://localhost:11434/v1".to_string(),
                api_key: String::new(),
                default_model: Some("llama3.2".to_string()),
            }),
        }
    }

    pub fn from_env() -> Result<Self> {
        Ok(Self::new())
    }
}

impl Default for OllamaProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Provider for OllamaProvider {
    async fn complete(&self, req: &ChatRequest) -> Result<ChatResponse> {
        self.inner.complete(req).await
    }

    async fn stream(&self, req: &ChatRequest) -> Result<BoxStream<'_, Result<StreamEvent>>> {
        self.inner.stream(req).await
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        self.inner.list_models().await
    }

    fn name(&self) -> &str {
        "ollama"
    }
}
