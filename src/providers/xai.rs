use async_trait::async_trait;
use futures::stream::BoxStream;

use crate::error::{BitAiError, Result};
use crate::model::ModelInfo;
use crate::provider::Provider;
use crate::providers::openai_compat::{OpenAICompatConfig, OpenAICompatProvider};
use crate::types::*;

pub struct XAIProvider {
    inner: OpenAICompatProvider,
}

impl XAIProvider {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            inner: OpenAICompatProvider::new(OpenAICompatConfig {
                name: "xai".to_string(),
                base_url: "https://api.x.ai/v1".to_string(),
                api_key: api_key.into(),
                default_model: Some("grok-3".to_string()),
            }),
        }
    }

    pub fn from_env() -> Result<Self> {
        let key = std::env::var("XAI_API_KEY")
            .map_err(|_| BitAiError::Config("XAI_API_KEY not set".to_string()))?;
        Ok(Self::new(key))
    }

    fn known_models() -> Vec<ModelInfo> {
        vec![
            ModelInfo {
                id: "grok-3".to_string(),
                provider: "xai".to_string(),
                display_name: Some("Grok 3".to_string()),
                context_window: Some(131_072),
                max_output_tokens: Some(131_072),
                supports_tools: true,
                supports_thinking: false,
                input_price_per_million: Some(3.00),
                output_price_per_million: Some(15.00),
            },
            ModelInfo {
                id: "grok-3-mini".to_string(),
                provider: "xai".to_string(),
                display_name: Some("Grok 3 Mini".to_string()),
                context_window: Some(131_072),
                max_output_tokens: Some(131_072),
                supports_tools: true,
                supports_thinking: true,
                input_price_per_million: Some(0.30),
                output_price_per_million: Some(0.50),
            },
        ]
    }
}

#[async_trait]
impl Provider for XAIProvider {
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
        "xai"
    }
}
