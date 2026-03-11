use std::sync::Arc;
use crate::model::register_provider;

/// Auto-register all providers that have API keys set in environment.
pub fn auto_register() {
    #[cfg(feature = "openai")]
    if let Ok(provider) = crate::providers::openai::OpenAIProvider::from_env() {
        register_provider(Arc::new(provider));
    }

    #[cfg(feature = "anthropic")]
    if let Ok(provider) = crate::providers::anthropic::AnthropicProvider::from_env() {
        register_provider(Arc::new(provider));
    }

    #[cfg(feature = "deepseek")]
    if let Ok(provider) = crate::providers::deepseek::DeepSeekProvider::from_env() {
        register_provider(Arc::new(provider));
    }

    #[cfg(feature = "groq")]
    if let Ok(provider) = crate::providers::groq::GroqProvider::from_env() {
        register_provider(Arc::new(provider));
    }

    #[cfg(feature = "xai")]
    if let Ok(provider) = crate::providers::xai::XAIProvider::from_env() {
        register_provider(Arc::new(provider));
    }

    #[cfg(feature = "mistral")]
    if let Ok(provider) = crate::providers::mistral::MistralProvider::from_env() {
        register_provider(Arc::new(provider));
    }

    #[cfg(feature = "ollama")]
    {
        let provider = crate::providers::ollama::OllamaProvider::from_env()
            .expect("OllamaProvider::from_env should never fail");
        register_provider(Arc::new(provider));
    }

    #[cfg(feature = "google")]
    if let Ok(provider) = crate::providers::google::GoogleProvider::from_env() {
        register_provider(Arc::new(provider));
    }
}
