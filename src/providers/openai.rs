use async_trait::async_trait;
use futures::stream::BoxStream;

use crate::error::{BitAiError, Result};
use crate::model::ModelInfo;
use crate::provider::Provider;
use crate::providers::openai_compat::{OpenAICompatConfig, OpenAICompatProvider};
use crate::types::*;

pub struct OpenAIProvider {
    inner: OpenAICompatProvider,
}

impl OpenAIProvider {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            inner: OpenAICompatProvider::new(OpenAICompatConfig {
                name: "openai".to_string(),
                base_url: "https://api.openai.com/v1".to_string(),
                api_key: api_key.into(),
                default_model: Some("gpt-4o".to_string()),
            }),
        }
    }

    pub fn from_env() -> Result<Self> {
        let key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| BitAiError::Config("OPENAI_API_KEY not set".to_string()))?;
        Ok(Self::new(key))
    }

    fn known_models() -> Vec<ModelInfo> {
        vec![
            ModelInfo {
                id: "gpt-4o".to_string(),
                provider: "openai".to_string(),
                display_name: Some("GPT-4o".to_string()),
                context_window: Some(128_000),
                max_output_tokens: Some(16_384),
                supports_tools: true,
                supports_thinking: false,
                input_price_per_million: Some(2.50),
                output_price_per_million: Some(10.00),
            },
            ModelInfo {
                id: "gpt-4o-mini".to_string(),
                provider: "openai".to_string(),
                display_name: Some("GPT-4o Mini".to_string()),
                context_window: Some(128_000),
                max_output_tokens: Some(16_384),
                supports_tools: true,
                supports_thinking: false,
                input_price_per_million: Some(0.15),
                output_price_per_million: Some(0.60),
            },
            ModelInfo {
                id: "o3".to_string(),
                provider: "openai".to_string(),
                display_name: Some("o3".to_string()),
                context_window: Some(200_000),
                max_output_tokens: Some(100_000),
                supports_tools: true,
                supports_thinking: true,
                input_price_per_million: Some(2.00),
                output_price_per_million: Some(8.00),
            },
            ModelInfo {
                id: "o3-mini".to_string(),
                provider: "openai".to_string(),
                display_name: Some("o3-mini".to_string()),
                context_window: Some(200_000),
                max_output_tokens: Some(100_000),
                supports_tools: true,
                supports_thinking: true,
                input_price_per_million: Some(1.10),
                output_price_per_million: Some(4.40),
            },
            ModelInfo {
                id: "o4-mini".to_string(),
                provider: "openai".to_string(),
                display_name: Some("o4-mini".to_string()),
                context_window: Some(200_000),
                max_output_tokens: Some(100_000),
                supports_tools: true,
                supports_thinking: true,
                input_price_per_million: Some(1.10),
                output_price_per_million: Some(4.40),
            },
        ]
    }
}

#[async_trait]
impl Provider for OpenAIProvider {
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
        "openai"
    }
}
