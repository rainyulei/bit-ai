pub mod openai_compat;

#[cfg(feature = "openai")]
pub mod openai;

#[cfg(feature = "anthropic")]
pub mod anthropic;

#[cfg(feature = "deepseek")]
pub mod deepseek;

#[cfg(feature = "groq")]
pub mod groq;

#[cfg(feature = "xai")]
pub mod xai;

#[cfg(feature = "mistral")]
pub mod mistral;

#[cfg(feature = "ollama")]
pub mod ollama;

#[cfg(feature = "google")]
pub mod google;
