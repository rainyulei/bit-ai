use async_trait::async_trait;
use futures::stream::BoxStream;

use crate::error::{BitAiError, Result};
use crate::model::ModelInfo;
use crate::provider::Provider;
use crate::providers::openai_compat::{OpenAICompatConfig, OpenAICompatProvider};
use crate::types::*;

pub struct GroqProvider {
    inner: OpenAICompatProvider,
}

impl GroqProvider {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            inner: OpenAICompatProvider::new(OpenAICompatConfig {
                name: "groq".to_string(),
                base_url: "https://api.groq.com/openai/v1".to_string(),
                api_key: api_key.into(),
                default_model: Some("llama-3.3-70b-versatile".to_string()),
            }),
        }
    }

    pub fn from_env() -> Result<Self> {
        let key = std::env::var("GROQ_API_KEY")
            .map_err(|_| BitAiError::Config("GROQ_API_KEY not set".to_string()))?;
        Ok(Self::new(key))
    }

    fn known_models() -> Vec<ModelInfo> {
        vec![
            ModelInfo {
                id: "llama-3.3-70b-versatile".to_string(),
                provider: "groq".to_string(),
                display_name: Some("Llama 3.3 70B Versatile".to_string()),
                context_window: Some(128_000),
                max_output_tokens: Some(32_768),
                supports_tools: true,
                supports_thinking: false,
                input_price_per_million: Some(0.59),
                output_price_per_million: Some(0.79),
            },
            ModelInfo {
                id: "llama-3.1-8b-instant".to_string(),
                provider: "groq".to_string(),
                display_name: Some("Llama 3.1 8B Instant".to_string()),
                context_window: Some(128_000),
                max_output_tokens: Some(8192),
                supports_tools: true,
                supports_thinking: false,
                input_price_per_million: Some(0.05),
                output_price_per_million: Some(0.08),
            },
            ModelInfo {
                id: "gemma2-9b-it".to_string(),
                provider: "groq".to_string(),
                display_name: Some("Gemma 2 9B IT".to_string()),
                context_window: Some(8192),
                max_output_tokens: Some(8192),
                supports_tools: true,
                supports_thinking: false,
                input_price_per_million: Some(0.20),
                output_price_per_million: Some(0.20),
            },
        ]
    }
}

#[async_trait]
impl Provider for GroqProvider {
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
        "groq"
    }
}
