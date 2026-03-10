# bit-ai Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build `bit-ai`, a Rust library crate providing a unified multi-provider LLM API with streaming, tool calling, thinking, usage tracking, and model discovery.

**Architecture:** Trait-based provider abstraction with top-level convenience functions. Each provider implements the `Provider` trait, dispatched via `Arc<dyn Provider>` at runtime. SSE parsing handled per-provider, normalized to unified `StreamEvent`.

**Tech Stack:** Rust, tokio, reqwest, serde, async-trait, futures, thiserror, tracing

---

### Task 1: Project Scaffold + Core Types

**Files:**
- Create: `Cargo.toml`
- Create: `src/lib.rs`
- Create: `src/error.rs`
- Create: `src/types.rs`

**Step 1: Initialize Cargo project**

Run: `cd /Users/rainlei/holiday/bit-ai-base && cargo init --lib --name bit-ai`

**Step 2: Set up Cargo.toml**

```toml
[package]
name = "bit-ai"
version = "0.1.0"
edition = "2024"
description = "Unified multi-provider LLM API for Rust"
license = "MIT"

[dependencies]
tokio = { version = "1", features = ["full"] }
reqwest = { version = "0.12", features = ["json", "stream"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
async-trait = "0.1"
futures = "0.3"
tokio-stream = "0.1"
thiserror = "2"
tokio-util = "0.7"
tracing = "0.1"
pin-project-lite = "0.2"

[features]
default = ["all-providers"]
all-providers = ["openai", "anthropic", "google", "mistral", "deepseek", "groq", "xai", "ollama"]
openai = []
anthropic = []
google = []
mistral = []
deepseek = []
groq = []
xai = []
ollama = []

[dev-dependencies]
tokio = { version = "1", features = ["full", "test-util"] }
```

**Step 3: Write error.rs**

```rust
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
```

**Step 4: Write types.rs**

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "thinking")]
    Thinking {
        content: String,
        signature: Option<String>,
    },
    #[serde(rename = "tool_call")]
    ToolCall {
        id: String,
        name: String,
        arguments: String,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_call_id: String,
        content: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentBlock>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<i64>,
}

impl Message {
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: vec![ContentBlock::Text { text: text.into() }],
            timestamp: None,
        }
    }

    pub fn system(text: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: vec![ContentBlock::Text { text: text.into() }],
            timestamp: None,
        }
    }

    pub fn assistant(text: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: vec![ContentBlock::Text { text: text.into() }],
            timestamp: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkingConfig {
    pub enabled: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub budget_tokens: Option<u32>,
}

#[derive(Debug, Clone, Default)]
pub struct ChatRequest {
    pub messages: Vec<Message>,
    pub tools: Option<Vec<ToolDef>>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub thinking: Option<ThinkingConfig>,
    pub stop: Option<Vec<String>>,
}

impl ChatRequest {
    pub fn new(messages: Vec<Message>) -> Self {
        Self {
            messages,
            ..Default::default()
        }
    }

    pub fn with_tools(mut self, tools: Vec<ToolDef>) -> Self {
        self.tools = Some(tools);
        self
    }

    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }

    pub fn with_max_tokens(mut self, max: u32) -> Self {
        self.max_tokens = Some(max);
        self
    }

    pub fn with_thinking(mut self, config: ThinkingConfig) -> Self {
        self.thinking = Some(config);
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Cost {
    pub input: f64,
    pub output: f64,
    pub total: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_write_tokens: Option<u32>,
    pub cost: Cost,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    pub message: Message,
    pub usage: Usage,
    pub model: String,
}

#[derive(Debug, Clone)]
pub enum StreamEvent {
    TextDelta(String),
    ThinkingDelta(String),
    ToolCallDelta {
        id: String,
        name: Option<String>,
        arguments_delta: String,
    },
    Usage(Usage),
    Done(ChatResponse),
    Error(crate::error::BitAiError),
}
```

**Step 5: Write initial lib.rs**

```rust
pub mod error;
pub mod types;

pub use error::{BitAiError, Result};
pub use types::*;
```

**Step 6: Verify it compiles**

Run: `cd /Users/rainlei/holiday/bit-ai-base && cargo check`
Expected: Compiles with no errors.

**Step 7: Commit**

```bash
git init
git add Cargo.toml src/
git commit -m "feat: project scaffold with core types (Message, ChatRequest, StreamEvent, Usage)"
```

---

### Task 2: Provider Trait + Model Registry

**Files:**
- Create: `src/provider.rs`
- Create: `src/model.rs`
- Modify: `src/lib.rs`

**Step 1: Write provider.rs**

```rust
use async_trait::async_trait;
use futures::stream::BoxStream;

use crate::error::Result;
use crate::model::ModelInfo;
use crate::types::{ChatRequest, ChatResponse, StreamEvent};

#[async_trait]
pub trait Provider: Send + Sync {
    async fn complete(&self, req: &ChatRequest) -> Result<ChatResponse>;

    async fn stream(&self, req: &ChatRequest) -> Result<BoxStream<'_, Result<StreamEvent>>>;

    async fn list_models(&self) -> Result<Vec<ModelInfo>>;

    fn name(&self) -> &str;
}
```

**Step 2: Write model.rs**

```rust
use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

use crate::error::{BitAiError, Result};
use crate::provider::Provider;

#[derive(Debug, Clone)]
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

pub struct Model {
    pub info: ModelInfo,
    pub provider: Arc<dyn Provider>,
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

pub fn model(provider_name: &str, model_id: &str) -> Result<Model> {
    let provider = get_provider(provider_name)?;
    let info = ModelInfo {
        id: model_id.to_string(),
        provider: provider_name.to_string(),
        display_name: None,
        context_window: None,
        max_output_tokens: None,
        supports_tools: true,
        supports_thinking: false,
        input_price_per_million: None,
        output_price_per_million: None,
    };
    Ok(Model { info, provider })
}
```

**Step 3: Update lib.rs with public API**

```rust
pub mod error;
pub mod model;
pub mod provider;
pub mod types;

pub use error::{BitAiError, Result};
pub use model::{Model, ModelInfo};
pub use provider::Provider;
pub use types::*;

use futures::stream::BoxStream;

/// Get a model handle by provider name and model ID.
pub fn model(provider: &str, model_id: &str) -> Result<Model> {
    model::model(provider, model_id)
}

/// Non-streaming chat completion.
pub async fn complete(model: &Model, req: ChatRequest) -> Result<ChatResponse> {
    model.provider.complete(&req).await
}

/// Streaming chat completion.
pub async fn stream(model: &Model, req: ChatRequest) -> Result<BoxStream<'_, Result<StreamEvent>>> {
    model.provider.stream(&req).await
}

/// List available models for a provider.
pub async fn list_models(provider: &str) -> Result<Vec<ModelInfo>> {
    let p = model::get_provider(provider)?;
    p.list_models().await
}

/// Register a custom provider.
pub fn register_provider(provider: std::sync::Arc<dyn Provider>) {
    model::register_provider(provider);
}
```

**Step 4: Verify it compiles**

Run: `cargo check`
Expected: Compiles with no errors.

**Step 5: Write basic test**

Create `tests/core_test.rs`:

```rust
use bit_ai::*;

#[test]
fn test_message_constructors() {
    let msg = Message::user("hello");
    assert_eq!(msg.role, Role::User);
    assert_eq!(msg.content.len(), 1);

    let msg = Message::system("you are helpful");
    assert_eq!(msg.role, Role::System);
}

#[test]
fn test_chat_request_builder() {
    let req = ChatRequest::new(vec![Message::user("hi")])
        .with_temperature(0.7)
        .with_max_tokens(1000);

    assert_eq!(req.temperature, Some(0.7));
    assert_eq!(req.max_tokens, Some(1000));
    assert_eq!(req.messages.len(), 1);
}

#[test]
fn test_unknown_provider() {
    let result = bit_ai::model("nonexistent", "model");
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        BitAiError::UnknownProvider(_)
    ));
}
```

**Step 6: Run tests**

Run: `cargo test`
Expected: All 3 tests pass.

**Step 7: Commit**

```bash
git add src/ tests/
git commit -m "feat: Provider trait, Model registry, and public API surface"
```

---

### Task 3: SSE Parser Utility

**Files:**
- Create: `src/sse.rs`
- Create: `tests/sse_test.rs`
- Modify: `src/lib.rs`

This is shared infrastructure used by all HTTP-based providers for parsing Server-Sent Events streams.

**Step 1: Write test for SSE parser**

Create `tests/sse_test.rs`:

```rust
use bit_ai::sse::{SseEvent, parse_sse_line};

#[test]
fn test_parse_data_line() {
    let mut event = SseEvent::default();
    parse_sse_line("data: {\"hello\":\"world\"}", &mut event);
    assert_eq!(event.data, Some("{\"hello\":\"world\"}".to_string()));
}

#[test]
fn test_parse_event_type() {
    let mut event = SseEvent::default();
    parse_sse_line("event: message_start", &mut event);
    assert_eq!(event.event, Some("message_start".to_string()));
}

#[test]
fn test_parse_done() {
    let mut event = SseEvent::default();
    parse_sse_line("data: [DONE]", &mut event);
    assert_eq!(event.data, Some("[DONE]".to_string()));
    assert!(event.is_done());
}

#[test]
fn test_ignore_comments_and_empty() {
    let mut event = SseEvent::default();
    parse_sse_line(": this is a comment", &mut event);
    assert!(event.data.is_none());
    parse_sse_line("", &mut event);
    assert!(event.data.is_none());
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test --test sse_test`
Expected: FAIL — module `sse` not found.

**Step 3: Implement sse.rs**

```rust
#[derive(Debug, Default, Clone)]
pub struct SseEvent {
    pub event: Option<String>,
    pub data: Option<String>,
    pub id: Option<String>,
}

impl SseEvent {
    pub fn is_done(&self) -> bool {
        self.data.as_deref() == Some("[DONE]")
    }

    pub fn reset(&mut self) {
        self.event = None;
        self.data = None;
        self.id = None;
    }
}

pub fn parse_sse_line(line: &str, event: &mut SseEvent) {
    if line.starts_with(':') || line.is_empty() {
        return;
    }

    if let Some(value) = line.strip_prefix("data: ") {
        event.data = Some(value.to_string());
    } else if let Some(value) = line.strip_prefix("event: ") {
        event.event = Some(value.to_string());
    } else if let Some(value) = line.strip_prefix("id: ") {
        event.id = Some(value.to_string());
    }
}
```

**Step 4: Add `pub mod sse;` to lib.rs**

**Step 5: Run tests**

Run: `cargo test --test sse_test`
Expected: All 4 tests pass.

**Step 6: Commit**

```bash
git add src/sse.rs tests/sse_test.rs src/lib.rs
git commit -m "feat: SSE parser utility for provider streaming"
```

---

### Task 4: OpenAI-Compatible Base Provider

**Files:**
- Create: `src/providers/mod.rs`
- Create: `src/providers/openai_compat.rs`
- Modify: `src/lib.rs`

This is the base implementation that OpenAI, DeepSeek, Groq, xAI, Ollama, and Mistral all build on top of. It handles message transformation to/from OpenAI format and SSE stream parsing.

**Step 1: Write providers/openai_compat.rs**

```rust
use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::error::{BitAiError, Result};
use crate::model::ModelInfo;
use crate::provider::Provider;
use crate::sse::{SseEvent, parse_sse_line};
use crate::types::*;

/// Configuration for an OpenAI-compatible provider.
#[derive(Debug, Clone)]
pub struct OpenAICompatConfig {
    pub name: String,
    pub base_url: String,
    pub api_key: String,
    pub default_model: Option<String>,
}

/// Generic OpenAI-compatible provider.
pub struct OpenAICompatProvider {
    config: OpenAICompatConfig,
    client: Client,
}

impl OpenAICompatProvider {
    pub fn new(config: OpenAICompatConfig) -> Self {
        Self {
            config,
            client: Client::new(),
        }
    }

    fn chat_url(&self) -> String {
        format!("{}/chat/completions", self.config.base_url.trim_end_matches('/'))
    }

    fn models_url(&self) -> String {
        format!("{}/models", self.config.base_url.trim_end_matches('/'))
    }

    fn build_request_body(&self, req: &ChatRequest, model_id: &str) -> serde_json::Value {
        let messages = convert_messages_to_openai(&req.messages);

        let mut body = json!({
            "model": model_id,
            "messages": messages,
        });

        if let Some(temp) = req.temperature {
            body["temperature"] = json!(temp);
        }
        if let Some(max) = req.max_tokens {
            body["max_tokens"] = json!(max);
        }
        if let Some(ref stop) = req.stop {
            body["stop"] = json!(stop);
        }
        if let Some(ref tools) = req.tools {
            body["tools"] = json!(tools.iter().map(|t| {
                json!({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    }
                })
            }).collect::<Vec<_>>());
        }

        body
    }
}

fn convert_messages_to_openai(messages: &[Message]) -> Vec<serde_json::Value> {
    messages.iter().map(|msg| {
        let role = match msg.role {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        };

        // Simple text-only messages
        if msg.content.len() == 1 {
            if let ContentBlock::Text { ref text } = msg.content[0] {
                return json!({ "role": role, "content": text });
            }
            if let ContentBlock::ToolResult { ref tool_call_id, ref content } = msg.content[0] {
                return json!({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": content,
                });
            }
        }

        // Assistant with tool calls
        if msg.role == Role::Assistant {
            let mut text_parts = Vec::new();
            let mut tool_calls = Vec::new();

            for block in &msg.content {
                match block {
                    ContentBlock::Text { text } => text_parts.push(text.clone()),
                    ContentBlock::ToolCall { id, name, arguments } => {
                        tool_calls.push(json!({
                            "id": id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": arguments,
                            }
                        }));
                    }
                    _ => {}
                }
            }

            let mut msg_json = json!({ "role": "assistant" });
            if !text_parts.is_empty() {
                msg_json["content"] = json!(text_parts.join(""));
            }
            if !tool_calls.is_empty() {
                msg_json["tool_calls"] = json!(tool_calls);
            }
            return msg_json;
        }

        // Fallback: join text blocks
        let text: String = msg.content.iter().filter_map(|b| {
            if let ContentBlock::Text { text } = b { Some(text.as_str()) } else { None }
        }).collect::<Vec<_>>().join("");

        json!({ "role": role, "content": text })
    }).collect()
}

fn parse_openai_response(body: serde_json::Value) -> Result<(Message, Usage, String)> {
    let model = body["model"].as_str().unwrap_or("unknown").to_string();
    let choice = &body["choices"][0];
    let msg = &choice["message"];

    let mut content = Vec::new();

    if let Some(text) = msg["content"].as_str() {
        if !text.is_empty() {
            content.push(ContentBlock::Text { text: text.to_string() });
        }
    }

    if let Some(tool_calls) = msg["tool_calls"].as_array() {
        for tc in tool_calls {
            content.push(ContentBlock::ToolCall {
                id: tc["id"].as_str().unwrap_or("").to_string(),
                name: tc["function"]["name"].as_str().unwrap_or("").to_string(),
                arguments: tc["function"]["arguments"].as_str().unwrap_or("{}").to_string(),
            });
        }
    }

    let usage_obj = &body["usage"];
    let input_tokens = usage_obj["prompt_tokens"].as_u64().unwrap_or(0) as u32;
    let output_tokens = usage_obj["completion_tokens"].as_u64().unwrap_or(0) as u32;

    let usage = Usage {
        input_tokens,
        output_tokens,
        ..Default::default()
    };

    let message = Message {
        role: Role::Assistant,
        content,
        timestamp: None,
    };

    Ok((message, usage, model))
}

#[async_trait]
impl Provider for OpenAICompatProvider {
    async fn complete(&self, req: &ChatRequest) -> Result<ChatResponse> {
        let model_id = self.config.default_model.as_deref().unwrap_or("gpt-4o");
        let mut body = self.build_request_body(req, model_id);
        body["stream"] = json!(false);

        let resp = self.client
            .post(&self.chat_url())
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?;

        let status = resp.status();
        if !status.is_success() {
            let error_text = resp.text().await.unwrap_or_default();
            return Err(BitAiError::Provider {
                status: status.as_u16(),
                message: error_text,
            });
        }

        let resp_body: serde_json::Value = resp.json().await?;
        let (message, usage, model) = parse_openai_response(resp_body)?;

        Ok(ChatResponse {
            message,
            usage,
            model,
        })
    }

    async fn stream(&self, req: &ChatRequest) -> Result<BoxStream<'_, Result<StreamEvent>>> {
        let model_id = self.config.default_model.as_deref().unwrap_or("gpt-4o");
        let mut body = self.build_request_body(req, model_id);
        body["stream"] = json!(true);

        let resp = self.client
            .post(&self.chat_url())
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?;

        let status = resp.status();
        if !status.is_success() {
            let error_text = resp.text().await.unwrap_or_default();
            return Err(BitAiError::Provider {
                status: status.as_u16(),
                message: error_text,
            });
        }

        let byte_stream = resp.bytes_stream();

        let stream = futures::stream::unfold(
            (byte_stream, String::new(), SseEvent::default(), Vec::new(), Usage::default()),
            |(mut byte_stream, mut buffer, mut sse_event, mut content_blocks, mut usage)| async move {
                use futures::StreamExt;

                loop {
                    // Try to extract a complete line from buffer
                    if let Some(newline_pos) = buffer.find('\n') {
                        let line = buffer[..newline_pos].trim_end_matches('\r').to_string();
                        buffer = buffer[newline_pos + 1..].to_string();

                        if line.is_empty() {
                            // Empty line = event boundary
                            if let Some(ref data) = sse_event.data {
                                if sse_event.is_done() {
                                    // Build final response
                                    let text: String = content_blocks.iter().filter_map(|b| {
                                        if let ContentBlock::Text { text } = b { Some(text.as_str()) } else { None }
                                    }).collect::<Vec<_>>().join("");

                                    let message = Message {
                                        role: Role::Assistant,
                                        content: if content_blocks.is_empty() {
                                            vec![ContentBlock::Text { text }]
                                        } else {
                                            content_blocks.clone()
                                        },
                                        timestamp: None,
                                    };

                                    let resp = ChatResponse {
                                        message,
                                        usage: usage.clone(),
                                        model: String::new(),
                                    };

                                    return Some((Ok(StreamEvent::Done(resp)), (byte_stream, buffer, sse_event, content_blocks, usage)));
                                }

                                if let Ok(chunk) = serde_json::from_str::<serde_json::Value>(data) {
                                    let events = parse_openai_stream_chunk(&chunk, &mut content_blocks, &mut usage);
                                    sse_event.reset();

                                    if let Some(event) = events {
                                        return Some((Ok(event), (byte_stream, buffer, sse_event, content_blocks, usage)));
                                    }
                                }
                            }
                            sse_event.reset();
                            continue;
                        }

                        parse_sse_line(&line, &mut sse_event);
                        continue;
                    }

                    // Need more data
                    match byte_stream.next().await {
                        Some(Ok(bytes)) => {
                            buffer.push_str(&String::from_utf8_lossy(&bytes));
                        }
                        Some(Err(e)) => {
                            return Some((Err(BitAiError::Http(e)), (byte_stream, buffer, sse_event, content_blocks, usage)));
                        }
                        None => return None,
                    }
                }
            },
        );

        Ok(Box::pin(stream))
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        let resp = self.client
            .get(&self.models_url())
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .send()
            .await?;

        let status = resp.status();
        if !status.is_success() {
            let error_text = resp.text().await.unwrap_or_default();
            return Err(BitAiError::Provider {
                status: status.as_u16(),
                message: error_text,
            });
        }

        let body: serde_json::Value = resp.json().await?;
        let models = body["data"]
            .as_array()
            .unwrap_or(&Vec::new())
            .iter()
            .filter_map(|m| {
                let id = m["id"].as_str()?.to_string();
                Some(ModelInfo {
                    id: id.clone(),
                    provider: self.config.name.clone(),
                    display_name: Some(id),
                    context_window: None,
                    max_output_tokens: None,
                    supports_tools: true,
                    supports_thinking: false,
                    input_price_per_million: None,
                    output_price_per_million: None,
                })
            })
            .collect();

        Ok(models)
    }

    fn name(&self) -> &str {
        &self.config.name
    }
}

fn parse_openai_stream_chunk(
    chunk: &serde_json::Value,
    content_blocks: &mut Vec<ContentBlock>,
    usage: &mut Usage,
) -> Option<StreamEvent> {
    let choice = &chunk["choices"][0];
    let delta = &choice["delta"];

    // Text delta
    if let Some(text) = delta["content"].as_str() {
        if !text.is_empty() {
            // Accumulate for final response
            let found = content_blocks.iter_mut().find(|b| matches!(b, ContentBlock::Text { .. }));
            if let Some(ContentBlock::Text { text: ref mut t }) = found {
                t.push_str(text);
            } else {
                content_blocks.push(ContentBlock::Text { text: text.to_string() });
            }
            return Some(StreamEvent::TextDelta(text.to_string()));
        }
    }

    // Tool call delta
    if let Some(tool_calls) = delta["tool_calls"].as_array() {
        for tc in tool_calls {
            let id = tc["id"].as_str().unwrap_or("").to_string();
            let name = tc["function"]["name"].as_str().map(|s| s.to_string());
            let args = tc["function"]["arguments"].as_str().unwrap_or("").to_string();

            if let Some(ref n) = name {
                content_blocks.push(ContentBlock::ToolCall {
                    id: id.clone(),
                    name: n.clone(),
                    arguments: args.clone(),
                });
            } else if !args.is_empty() {
                // Append arguments to last tool call
                if let Some(ContentBlock::ToolCall { arguments, .. }) = content_blocks.last_mut() {
                    arguments.push_str(&args);
                }
            }

            return Some(StreamEvent::ToolCallDelta {
                id,
                name,
                arguments_delta: args,
            });
        }
    }

    // Usage in stream
    if let Some(u) = chunk.get("usage") {
        usage.input_tokens = u["prompt_tokens"].as_u64().unwrap_or(0) as u32;
        usage.output_tokens = u["completion_tokens"].as_u64().unwrap_or(0) as u32;
        return Some(StreamEvent::Usage(usage.clone()));
    }

    None
}
```

**Step 2: Write providers/mod.rs**

```rust
pub mod openai_compat;
```

**Step 3: Add `pub mod providers;` to lib.rs**

**Step 4: Verify it compiles**

Run: `cargo check`
Expected: Compiles.

**Step 5: Commit**

```bash
git add src/providers/
git commit -m "feat: OpenAI-compatible base provider with streaming SSE and tool calling"
```

---

### Task 5: OpenAI Provider

**Files:**
- Create: `src/providers/openai.rs`
- Modify: `src/providers/mod.rs`

**Step 1: Write openai.rs**

Uses `OpenAICompatProvider` internally with OpenAI-specific defaults and model metadata.

```rust
use std::sync::Arc;
use crate::error::Result;
use crate::model::ModelInfo;
use crate::providers::openai_compat::{OpenAICompatConfig, OpenAICompatProvider};
use crate::provider::Provider;
use crate::types::*;
use async_trait::async_trait;
use futures::stream::BoxStream;

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
            .map_err(|_| crate::error::BitAiError::Config("OPENAI_API_KEY not set".to_string()))?;
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
        // Return known models with pricing, fall back to API discovery
        Ok(Self::known_models())
    }

    fn name(&self) -> &str {
        "openai"
    }
}
```

**Step 2: Add to providers/mod.rs**

```rust
pub mod openai_compat;
#[cfg(feature = "openai")]
pub mod openai;
```

**Step 3: Verify**

Run: `cargo check`

**Step 4: Commit**

```bash
git add src/providers/openai.rs src/providers/mod.rs
git commit -m "feat: OpenAI provider with known model metadata and pricing"
```

---

### Task 6: Anthropic Provider

**Files:**
- Create: `src/providers/anthropic.rs`
- Modify: `src/providers/mod.rs`

Anthropic uses a different API format (Messages API), so it gets its own implementation rather than reusing `openai_compat`.

**Step 1: Write anthropic.rs**

Anthropic specifics:
- Endpoint: `https://api.anthropic.com/v1/messages`
- Auth header: `x-api-key` (not Bearer)
- Header: `anthropic-version: 2023-06-01`
- Different message format: `content` is array of blocks
- Thinking support with `thinking` parameter
- SSE events: `message_start`, `content_block_start`, `content_block_delta`, `content_block_stop`, `message_delta`, `message_stop`

```rust
use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::StreamExt;
use reqwest::Client;
use serde_json::json;

use crate::error::{BitAiError, Result};
use crate::model::ModelInfo;
use crate::provider::Provider;
use crate::sse::{SseEvent, parse_sse_line};
use crate::types::*;

pub struct AnthropicProvider {
    api_key: String,
    base_url: String,
    client: Client,
}

impl AnthropicProvider {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: "https://api.anthropic.com".to_string(),
            client: Client::new(),
        }
    }

    pub fn from_env() -> Result<Self> {
        let key = std::env::var("ANTHROPIC_API_KEY")
            .map_err(|_| BitAiError::Config("ANTHROPIC_API_KEY not set".to_string()))?;
        Ok(Self::new(key))
    }

    fn messages_url(&self) -> String {
        format!("{}/v1/messages", self.base_url)
    }

    fn build_request_body(&self, req: &ChatRequest, model_id: &str) -> serde_json::Value {
        let (system, messages) = convert_messages_to_anthropic(&req.messages);

        let mut body = json!({
            "model": model_id,
            "messages": messages,
            "max_tokens": req.max_tokens.unwrap_or(4096),
        });

        if let Some(ref sys) = system {
            body["system"] = json!(sys);
        }
        if let Some(temp) = req.temperature {
            body["temperature"] = json!(temp);
        }
        if let Some(ref stop) = req.stop {
            body["stop_sequences"] = json!(stop);
        }
        if let Some(ref tools) = req.tools {
            body["tools"] = json!(tools.iter().map(|t| {
                json!({
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters,
                })
            }).collect::<Vec<_>>());
        }
        if let Some(ref thinking) = req.thinking {
            if thinking.enabled {
                body["thinking"] = json!({
                    "type": "enabled",
                    "budget_tokens": thinking.budget_tokens.unwrap_or(10000),
                });
            }
        }

        body
    }

    fn known_models() -> Vec<ModelInfo> {
        vec![
            ModelInfo {
                id: "claude-opus-4-20250514".to_string(),
                provider: "anthropic".to_string(),
                display_name: Some("Claude Opus 4".to_string()),
                context_window: Some(200_000),
                max_output_tokens: Some(32_000),
                supports_tools: true,
                supports_thinking: true,
                input_price_per_million: Some(15.00),
                output_price_per_million: Some(75.00),
            },
            ModelInfo {
                id: "claude-sonnet-4-20250514".to_string(),
                provider: "anthropic".to_string(),
                display_name: Some("Claude Sonnet 4".to_string()),
                context_window: Some(200_000),
                max_output_tokens: Some(16_000),
                supports_tools: true,
                supports_thinking: true,
                input_price_per_million: Some(3.00),
                output_price_per_million: Some(15.00),
            },
            ModelInfo {
                id: "claude-haiku-4-20250514".to_string(),
                provider: "anthropic".to_string(),
                display_name: Some("Claude Haiku 4".to_string()),
                context_window: Some(200_000),
                max_output_tokens: Some(8_000),
                supports_tools: true,
                supports_thinking: true,
                input_price_per_million: Some(0.80),
                output_price_per_million: Some(4.00),
            },
        ]
    }
}

fn convert_messages_to_anthropic(messages: &[Message]) -> (Option<String>, Vec<serde_json::Value>) {
    let mut system = None;
    let mut result = Vec::new();

    for msg in messages {
        match msg.role {
            Role::System => {
                let text: String = msg.content.iter().filter_map(|b| {
                    if let ContentBlock::Text { text } = b { Some(text.as_str()) } else { None }
                }).collect::<Vec<_>>().join("\n");
                system = Some(text);
            }
            Role::User => {
                let content: Vec<serde_json::Value> = msg.content.iter().map(|b| {
                    match b {
                        ContentBlock::Text { text } => json!({ "type": "text", "text": text }),
                        ContentBlock::ToolResult { tool_call_id, content } => json!({
                            "type": "tool_result",
                            "tool_use_id": tool_call_id,
                            "content": content,
                        }),
                        _ => json!({ "type": "text", "text": "" }),
                    }
                }).collect();
                result.push(json!({ "role": "user", "content": content }));
            }
            Role::Assistant => {
                let content: Vec<serde_json::Value> = msg.content.iter().filter_map(|b| {
                    match b {
                        ContentBlock::Text { text } => Some(json!({ "type": "text", "text": text })),
                        ContentBlock::Thinking { content, signature } => Some(json!({
                            "type": "thinking",
                            "thinking": content,
                            "signature": signature,
                        })),
                        ContentBlock::ToolCall { id, name, arguments } => {
                            let input: serde_json::Value = serde_json::from_str(arguments).unwrap_or(json!({}));
                            Some(json!({
                                "type": "tool_use",
                                "id": id,
                                "name": name,
                                "input": input,
                            }))
                        }
                        _ => None,
                    }
                }).collect();
                result.push(json!({ "role": "assistant", "content": content }));
            }
            Role::Tool => {
                // Tool results are sent as user messages in Anthropic
                let content: Vec<serde_json::Value> = msg.content.iter().filter_map(|b| {
                    if let ContentBlock::ToolResult { tool_call_id, content } = b {
                        Some(json!({
                            "type": "tool_result",
                            "tool_use_id": tool_call_id,
                            "content": content,
                        }))
                    } else {
                        None
                    }
                }).collect();
                result.push(json!({ "role": "user", "content": content }));
            }
        }
    }

    (system, result)
}

#[async_trait]
impl Provider for AnthropicProvider {
    async fn complete(&self, req: &ChatRequest) -> Result<ChatResponse> {
        let model_id = "claude-sonnet-4-20250514";
        let body = self.build_request_body(req, model_id);

        let resp = self.client
            .post(&self.messages_url())
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?;

        let status = resp.status();
        if !status.is_success() {
            let error_text = resp.text().await.unwrap_or_default();
            return Err(BitAiError::Provider {
                status: status.as_u16(),
                message: error_text,
            });
        }

        let resp_body: serde_json::Value = resp.json().await?;
        let message = parse_anthropic_response(&resp_body);
        let usage = parse_anthropic_usage(&resp_body);
        let model = resp_body["model"].as_str().unwrap_or(model_id).to_string();

        Ok(ChatResponse { message, usage, model })
    }

    async fn stream(&self, req: &ChatRequest) -> Result<BoxStream<'_, Result<StreamEvent>>> {
        let model_id = "claude-sonnet-4-20250514";
        let mut body = self.build_request_body(req, model_id);
        body["stream"] = json!(true);

        let resp = self.client
            .post(&self.messages_url())
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?;

        let status = resp.status();
        if !status.is_success() {
            let error_text = resp.text().await.unwrap_or_default();
            return Err(BitAiError::Provider {
                status: status.as_u16(),
                message: error_text,
            });
        }

        let byte_stream = resp.bytes_stream();

        let stream = futures::stream::unfold(
            (byte_stream, String::new(), SseEvent::default(), Vec::<ContentBlock>::new(), Usage::default()),
            |(mut byte_stream, mut buffer, mut sse_event, mut content_blocks, mut usage)| async move {
                loop {
                    if let Some(newline_pos) = buffer.find('\n') {
                        let line = buffer[..newline_pos].trim_end_matches('\r').to_string();
                        buffer = buffer[newline_pos + 1..].to_string();

                        if line.is_empty() {
                            if let Some(ref data) = sse_event.data {
                                if let Ok(chunk) = serde_json::from_str::<serde_json::Value>(data) {
                                    let event_type = sse_event.event.as_deref().unwrap_or("");
                                    let evt = parse_anthropic_stream_event(event_type, &chunk, &mut content_blocks, &mut usage);
                                    sse_event.reset();
                                    if let Some(event) = evt {
                                        return Some((Ok(event), (byte_stream, buffer, sse_event, content_blocks, usage)));
                                    }
                                }
                            }
                            sse_event.reset();
                            continue;
                        }

                        parse_sse_line(&line, &mut sse_event);
                        continue;
                    }

                    match byte_stream.next().await {
                        Some(Ok(bytes)) => {
                            buffer.push_str(&String::from_utf8_lossy(&bytes));
                        }
                        Some(Err(e)) => {
                            return Some((Err(BitAiError::Http(e)), (byte_stream, buffer, sse_event, content_blocks, usage)));
                        }
                        None => return None,
                    }
                }
            },
        );

        Ok(Box::pin(stream))
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        Ok(Self::known_models())
    }

    fn name(&self) -> &str {
        "anthropic"
    }
}

fn parse_anthropic_response(body: &serde_json::Value) -> Message {
    let mut content = Vec::new();

    if let Some(blocks) = body["content"].as_array() {
        for block in blocks {
            match block["type"].as_str() {
                Some("text") => {
                    if let Some(text) = block["text"].as_str() {
                        content.push(ContentBlock::Text { text: text.to_string() });
                    }
                }
                Some("thinking") => {
                    content.push(ContentBlock::Thinking {
                        content: block["thinking"].as_str().unwrap_or("").to_string(),
                        signature: block["signature"].as_str().map(|s| s.to_string()),
                    });
                }
                Some("tool_use") => {
                    content.push(ContentBlock::ToolCall {
                        id: block["id"].as_str().unwrap_or("").to_string(),
                        name: block["name"].as_str().unwrap_or("").to_string(),
                        arguments: block["input"].to_string(),
                    });
                }
                _ => {}
            }
        }
    }

    Message {
        role: Role::Assistant,
        content,
        timestamp: None,
    }
}

fn parse_anthropic_usage(body: &serde_json::Value) -> Usage {
    let u = &body["usage"];
    Usage {
        input_tokens: u["input_tokens"].as_u64().unwrap_or(0) as u32,
        output_tokens: u["output_tokens"].as_u64().unwrap_or(0) as u32,
        cache_read_tokens: u["cache_read_input_tokens"].as_u64().map(|v| v as u32),
        cache_write_tokens: u["cache_creation_input_tokens"].as_u64().map(|v| v as u32),
        ..Default::default()
    }
}

fn parse_anthropic_stream_event(
    event_type: &str,
    data: &serde_json::Value,
    content_blocks: &mut Vec<ContentBlock>,
    usage: &mut Usage,
) -> Option<StreamEvent> {
    match event_type {
        "message_start" => {
            if let Some(u) = data.get("message").and_then(|m| m.get("usage")) {
                usage.input_tokens = u["input_tokens"].as_u64().unwrap_or(0) as u32;
            }
            None
        }
        "content_block_start" => {
            let block = &data["content_block"];
            match block["type"].as_str() {
                Some("text") => {
                    content_blocks.push(ContentBlock::Text { text: String::new() });
                }
                Some("thinking") => {
                    content_blocks.push(ContentBlock::Thinking {
                        content: String::new(),
                        signature: None,
                    });
                }
                Some("tool_use") => {
                    content_blocks.push(ContentBlock::ToolCall {
                        id: block["id"].as_str().unwrap_or("").to_string(),
                        name: block["name"].as_str().unwrap_or("").to_string(),
                        arguments: String::new(),
                    });
                }
                _ => {}
            }
            None
        }
        "content_block_delta" => {
            let delta = &data["delta"];
            match delta["type"].as_str() {
                Some("text_delta") => {
                    let text = delta["text"].as_str().unwrap_or("").to_string();
                    if let Some(ContentBlock::Text { text: ref mut t }) = content_blocks.last_mut() {
                        t.push_str(&text);
                    }
                    Some(StreamEvent::TextDelta(text))
                }
                Some("thinking_delta") => {
                    let thinking = delta["thinking"].as_str().unwrap_or("").to_string();
                    if let Some(ContentBlock::Thinking { ref mut content, .. }) = content_blocks.last_mut() {
                        content.push_str(&thinking);
                    }
                    Some(StreamEvent::ThinkingDelta(thinking))
                }
                Some("input_json_delta") => {
                    let partial = delta["partial_json"].as_str().unwrap_or("").to_string();
                    let (id, name) = if let Some(ContentBlock::ToolCall { id, name, ref mut arguments }) = content_blocks.last_mut() {
                        arguments.push_str(&partial);
                        (id.clone(), Some(name.clone()))
                    } else {
                        (String::new(), None)
                    };
                    Some(StreamEvent::ToolCallDelta {
                        id,
                        name,
                        arguments_delta: partial,
                    })
                }
                Some("signature_delta") => {
                    let sig = delta["signature"].as_str().unwrap_or("").to_string();
                    if let Some(ContentBlock::Thinking { ref mut signature, .. }) = content_blocks.last_mut() {
                        *signature = Some(sig);
                    }
                    None
                }
                _ => None,
            }
        }
        "message_delta" => {
            if let Some(u) = data.get("usage") {
                usage.output_tokens = u["output_tokens"].as_u64().unwrap_or(0) as u32;
                return Some(StreamEvent::Usage(usage.clone()));
            }
            None
        }
        "message_stop" => {
            let message = Message {
                role: Role::Assistant,
                content: content_blocks.clone(),
                timestamp: None,
            };
            Some(StreamEvent::Done(ChatResponse {
                message,
                usage: usage.clone(),
                model: String::new(),
            }))
        }
        _ => None,
    }
}
```

**Step 2: Add to providers/mod.rs**

```rust
#[cfg(feature = "anthropic")]
pub mod anthropic;
```

**Step 3: Verify**

Run: `cargo check`

**Step 4: Commit**

```bash
git add src/providers/anthropic.rs src/providers/mod.rs
git commit -m "feat: Anthropic provider with Messages API, thinking, and tool calling"
```

---

### Task 7: Remaining Providers (thin wrappers)

**Files:**
- Create: `src/providers/deepseek.rs`
- Create: `src/providers/groq.rs`
- Create: `src/providers/xai.rs`
- Create: `src/providers/mistral.rs`
- Create: `src/providers/ollama.rs`
- Create: `src/providers/google.rs`
- Modify: `src/providers/mod.rs`

These are all OpenAI-compatible (except Google) and follow the same pattern as OpenAI — thin wrappers around `OpenAICompatProvider` with different base URLs and known models.

**Step 1: Write deepseek.rs**

```rust
use std::sync::Arc;
use async_trait::async_trait;
use futures::stream::BoxStream;

use crate::error::Result;
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
            .map_err(|_| crate::error::BitAiError::Config("DEEPSEEK_API_KEY not set".to_string()))?;
        Ok(Self::new(key))
    }
}

#[async_trait]
impl Provider for DeepSeekProvider {
    async fn complete(&self, req: &ChatRequest) -> Result<ChatResponse> { self.inner.complete(req).await }
    async fn stream(&self, req: &ChatRequest) -> Result<BoxStream<'_, Result<StreamEvent>>> { self.inner.stream(req).await }
    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        Ok(vec![
            ModelInfo {
                id: "deepseek-chat".to_string(),
                provider: "deepseek".to_string(),
                display_name: Some("DeepSeek Chat (V3)".to_string()),
                context_window: Some(64_000),
                max_output_tokens: Some(8_192),
                supports_tools: true,
                supports_thinking: false,
                input_price_per_million: Some(0.27),
                output_price_per_million: Some(1.10),
            },
            ModelInfo {
                id: "deepseek-reasoner".to_string(),
                provider: "deepseek".to_string(),
                display_name: Some("DeepSeek Reasoner (R1)".to_string()),
                context_window: Some(64_000),
                max_output_tokens: Some(8_192),
                supports_tools: false,
                supports_thinking: true,
                input_price_per_million: Some(0.55),
                output_price_per_million: Some(2.19),
            },
        ])
    }
    fn name(&self) -> &str { "deepseek" }
}
```

**Step 2: Write groq.rs, xai.rs, mistral.rs, ollama.rs** — same pattern, different `base_url`, `env var`, and `known_models()`. (Full code provided in each file following the DeepSeek template.)

**Step 3: Write google.rs** — Google Gemini has its own API format (`generateContent` endpoint), similar to Anthropic it gets a standalone implementation.

**Step 4: Update providers/mod.rs with all feature-gated modules**

**Step 5: Verify**

Run: `cargo check`

**Step 6: Commit**

```bash
git add src/providers/
git commit -m "feat: add DeepSeek, Groq, xAI, Mistral, Ollama, and Google providers"
```

---

### Task 8: Auto-Registration + Integration Test

**Files:**
- Create: `src/registry.rs` — auto-register providers from env vars
- Create: `tests/integration_test.rs`
- Modify: `src/lib.rs`

**Step 1: Write registry.rs**

```rust
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

    // ... same for groq, xai, mistral, ollama, google
}
```

**Step 2: Add `init()` to lib.rs**

```rust
pub mod registry;

/// Initialize bit-ai: auto-register providers from environment variables.
pub fn init() {
    registry::auto_register();
}
```

**Step 3: Write integration test** (requires real API keys, marked `#[ignore]`)

```rust
use bit_ai::*;
use futures::StreamExt;

#[tokio::test]
#[ignore] // Requires OPENAI_API_KEY
async fn test_openai_complete() {
    bit_ai::init();
    let model = bit_ai::model("openai", "gpt-4o-mini").unwrap();
    let req = ChatRequest::new(vec![Message::user("Say 'hello' and nothing else.")]);
    let resp = bit_ai::complete(&model, req).await.unwrap();

    assert!(!resp.message.content.is_empty());
    assert!(resp.usage.input_tokens > 0);
}

#[tokio::test]
#[ignore] // Requires OPENAI_API_KEY
async fn test_openai_stream() {
    bit_ai::init();
    let model = bit_ai::model("openai", "gpt-4o-mini").unwrap();
    let req = ChatRequest::new(vec![Message::user("Say 'hello' and nothing else.")]);
    let mut stream = bit_ai::stream(&model, req).await.unwrap();

    let mut got_text = false;
    while let Some(event) = stream.next().await {
        match event.unwrap() {
            StreamEvent::TextDelta(_) => got_text = true,
            StreamEvent::Done(resp) => {
                assert!(!resp.message.content.is_empty());
                break;
            }
            _ => {}
        }
    }
    assert!(got_text);
}

#[tokio::test]
#[ignore] // Requires ANTHROPIC_API_KEY
async fn test_anthropic_complete() {
    bit_ai::init();
    let model = bit_ai::model("anthropic", "claude-haiku-4-20250514").unwrap();
    let req = ChatRequest::new(vec![Message::user("Say 'hello' and nothing else.")]);
    let resp = bit_ai::complete(&model, req).await.unwrap();

    assert!(!resp.message.content.is_empty());
}
```

**Step 4: Run unit tests (non-ignored)**

Run: `cargo test`
Expected: All existing tests pass.

**Step 5: Commit**

```bash
git add src/registry.rs tests/integration_test.rs src/lib.rs
git commit -m "feat: auto-registration from env vars and integration tests"
```

---

### Task 9: Cost Calculation

**Files:**
- Modify: `src/providers/openai.rs` — add cost calculation to `complete`
- Modify: `src/providers/anthropic.rs` — add cost calculation
- Modify: `src/model.rs` — add `calculate_cost` helper

**Step 1: Add `calculate_cost` to model.rs**

```rust
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
```

**Step 2: Wire cost calculation into providers' `complete()` and streaming `Done` events**

**Step 3: Write test**

```rust
#[test]
fn test_cost_calculation() {
    let info = ModelInfo {
        id: "test".to_string(),
        provider: "test".to_string(),
        input_price_per_million: Some(3.0),
        output_price_per_million: Some(15.0),
        ..Default::default()
    };

    let usage = Usage {
        input_tokens: 1000,
        output_tokens: 500,
        ..Default::default()
    };

    let cost = info.calculate_cost(&usage);
    assert!((cost.input - 0.003).abs() < 0.0001);
    assert!((cost.output - 0.0075).abs() < 0.0001);
    assert!((cost.total - 0.0105).abs() < 0.0001);
}
```

**Step 4: Run tests**

Run: `cargo test`

**Step 5: Commit**

```bash
git add src/model.rs src/providers/
git commit -m "feat: cost calculation based on model pricing metadata"
```

---

## Summary

| Task | Description | Est. |
|------|------------|------|
| 1 | Project scaffold + core types | 5 min |
| 2 | Provider trait + model registry + public API | 5 min |
| 3 | SSE parser utility | 3 min |
| 4 | OpenAI-compatible base provider | 10 min |
| 5 | OpenAI provider | 3 min |
| 6 | Anthropic provider | 10 min |
| 7 | Remaining providers (DeepSeek, Groq, xAI, Mistral, Ollama, Google) | 15 min |
| 8 | Auto-registration + integration tests | 5 min |
| 9 | Cost calculation | 5 min |
