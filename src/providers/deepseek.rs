use async_trait::async_trait;
use futures::stream::BoxStream;

use crate::error::{BitAiError, Result};
use crate::model::ModelInfo;
use crate::provider::Provider;
use crate::providers::openai_compat::{OpenAICompatConfig, OpenAICompatProvider};
use crate::types::*;

pub struct DeepSeekProvider {
    inner: OpenAICompatProvider,
}

impl DeepSeekProvider {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            inner: OpenAICompatProvider::new(OpenAICompatConfig {
                name: "deepseek".to_string(),
                base_url: "https://api.deepseek.com/v1".to_string(),
                api_key: api_key.into(),
                default_model: Some("deepseek-chat".to_string()),
            }),
        }
    }

    pub fn from_env() -> Result<Self> {
        let key = std::env::var("DEEPSEEK_API_KEY")
            .map_err(|_| BitAiError::Config("DEEPSEEK_API_KEY not set".to_string()))?;
        Ok(Self::new(key))
    }

    fn known_models() -> Vec<ModelInfo> {
        vec![
            ModelInfo {
                id: "deepseek-chat".to_string(),
                provider: "deepseek".to_string(),
                display_name: Some("DeepSeek Chat".to_string()),
                context_window: Some(64_000),
                max_output_tokens: Some(8192),
                supports_tools: true,
                supports_thinking: false,
                input_price_per_million: Some(0.27),
                output_price_per_million: Some(1.10),
            },
            ModelInfo {
                id: "deepseek-reasoner".to_string(),
                provider: "deepseek".to_string(),
                display_name: Some("DeepSeek Reasoner".to_string()),
                context_window: Some(64_000),
                max_output_tokens: Some(8192),
                supports_tools: false,
                supports_thinking: true,
                input_price_per_million: Some(0.55),
                output_price_per_million: Some(2.19),
            },
        ]
    }
}

#[async_trait]
impl Provider for DeepSeekProvider {
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
        "deepseek"
    }
}
