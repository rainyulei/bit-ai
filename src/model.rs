use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

use crate::error::{BitAiError, Result};
use crate::provider::Provider;
use crate::types::Cost;
use crate::types::Usage;

#[derive(Debug, Clone, Default)]
pub struct ModelInfo {
    pub id: String,
    pub provider: String,
    pub display_name: Option<String>,
    pub context_window: Option<u32>,
    pub max_output_tokens: Option<u32>,
    pub supports_tools: bool,
    pub supports_thinking: bool,
    pub input_price_per_million: Option<f64>,
    pub output_price_per_million: Option<f64>,
}

impl ModelInfo {
    pub fn calculate_cost(&self, usage: &Usage) -> Cost {
        let input_rate = self.input_price_per_million.unwrap_or(0.0) / 1_000_000.0;
        let output_rate = self.output_price_per_million.unwrap_or(0.0) / 1_000_000.0;

        let input_cost = usage.input_tokens as f64 * input_rate;
        let output_cost = usage.output_tokens as f64 * output_rate;

        Cost {
            input: input_cost,
            output: output_cost,
            total: input_cost + output_cost,
        }
    }
}

pub struct Model {
    pub info: ModelInfo,
    pub provider: Arc<dyn Provider>,
}

impl std::fmt::Debug for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Model")
            .field("info", &self.info)
            .field("provider", &self.provider.name())
            .finish()
    }
}

type ProviderRegistry = HashMap<String, Arc<dyn Provider>>;

fn global_registry() -> &'static RwLock<ProviderRegistry> {
    static REGISTRY: OnceLock<RwLock<ProviderRegistry>> = OnceLock::new();
    REGISTRY.get_or_init(|| RwLock::new(HashMap::new()))
}

pub fn register_provider(provider: Arc<dyn Provider>) {
    let name = provider.name().to_string();
    let mut registry = global_registry().write().unwrap();
    registry.insert(name, provider);
}

pub fn get_provider(name: &str) -> Result<Arc<dyn Provider>> {
    let registry = global_registry().read().unwrap();
    registry
        .get(name)
        .cloned()
        .ok_or_else(|| BitAiError::UnknownProvider(name.to_string()))
}

pub fn get_model(provider_name: &str, model_id: &str) -> Result<Model> {
    let provider = get_provider(provider_name)?;
    let info = ModelInfo {
        id: model_id.to_string(),
        provider: provider_name.to_string(),
        ..Default::default()
    };
    Ok(Model { info, provider })
}
