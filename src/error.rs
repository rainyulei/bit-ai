use thiserror::Error;

#[derive(Error, Debug)]
pub enum BitAiError {
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),

    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Provider error: {status} - {message}")]
    Provider { status: u16, message: String },

    #[error("Unknown provider: {0}")]
    UnknownProvider(String),

    #[error("Unknown model: {provider}/{model}")]
    UnknownModel { provider: String, model: String },

    #[error("Stream error: {0}")]
    Stream(String),

    #[error("Configuration error: {0}")]
    Config(String),
}

pub type Result<T> = std::result::Result<T, BitAiError>;
