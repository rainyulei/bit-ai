use async_trait::async_trait;
use futures::stream::BoxStream;

use crate::error::{BitAiError, Result};
use crate::model::ModelInfo;
use crate::provider::Provider;
use crate::providers::openai_compat::{OpenAICompatConfig, OpenAICompatProvider};
use crate::types::*;

pub struct MistralProvider {
    inner: OpenAICompatProvider,
}

impl MistralProvider {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            inner: OpenAICompatProvider::new(OpenAICompatConfig {
                name: "mistral".to_string(),
                base_url: "https://api.mistral.ai/v1".to_string(),
                api_key: api_key.into(),
                default_model: Some("mistral-large-latest".to_string()),
            }),
        }
    }

    pub fn from_env() -> Result<Self> {
        let key = std::env::var("MISTRAL_API_KEY")
            .map_err(|_| BitAiError::Config("MISTRAL_API_KEY not set".to_string()))?;
        Ok(Self::new(key))
    }

    fn known_models() -> Vec<ModelInfo> {
        vec![
            ModelInfo {
                id: "mistral-large-latest".to_string(),
                provider: "mistral".to_string(),
                display_name: Some("Mistral Large".to_string()),
                context_window: Some(128_000),
                max_output_tokens: Some(8192),
                supports_tools: true,
                supports_thinking: false,
                input_price_per_million: Some(2.00),
                output_price_per_million: Some(6.00),
            },
            ModelInfo {
                id: "mistral-small-latest".to_string(),
                provider: "mistral".to_string(),
                display_name: Some("Mistral Small".to_string()),
                context_window: Some(128_000),
                max_output_tokens: Some(8192),
                supports_tools: true,
                supports_thinking: false,
                input_price_per_million: Some(0.10),
                output_price_per_million: Some(0.30),
            },
            ModelInfo {
                id: "codestral-latest".to_string(),
                provider: "mistral".to_string(),
                display_name: Some("Codestral".to_string()),
                context_window: Some(256_000),
                max_output_tokens: Some(8192),
                supports_tools: true,
                supports_thinking: false,
                input_price_per_million: Some(0.30),
                output_price_per_million: Some(0.90),
            },
        ]
    }
}

#[async_trait]
impl Provider for MistralProvider {
    async fn complete(&self, req: &ChatRequest) -> Result<ChatResponse> {
        self.inner.complete(req).await
    }

    async fn stream(&self, req: &ChatRequest) -> Result<BoxStream<'_, Result<StreamEvent>>> {
        self.inner.stream(req).await
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        Ok(Self::known_models())
    }

    fn name(&self) -> &str {
        "mistral"
    }
}
