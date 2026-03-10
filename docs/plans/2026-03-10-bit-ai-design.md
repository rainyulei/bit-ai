# bit-ai Design Document

## Overview

`bit-ai` is a Rust library crate that provides a unified multi-provider LLM API. It serves as the foundational layer for higher-level tools (e.g., coding agents) to communicate with AI models across different providers through a single, consistent interface.

## Scope

### Included
- Multi-provider adapter: OpenAI, Anthropic, Gemini, Mistral, DeepSeek, Groq, xAI, Ollama, OpenAI-compatible
- Chat Completion: streaming + non-streaming
- Tool Calling
- Thinking/Reasoning support
- Token usage & cost tracking
- Auto model discovery (list available models from providers)
- Custom provider registration

### Excluded (future, separate crates)
- Context serialization / persistence
- Image input (vision)
- Voice / Realtime API

## Architecture

**Hybrid approach**: Trait abstraction at the bottom, convenience API at the top.

```
User Code
    ↓
bit_ai::model() / complete() / stream()    ← Public API (Box<dyn Provider>)
    ↓
Provider trait                               ← Abstraction layer
    ↓
openai.rs / anthropic.rs / google.rs ...    ← Concrete implementations
    ↓
reqwest HTTP / SSE                           ← Transport
```

## Project Structure

```
bit-ai/
├── Cargo.toml
├── src/
│   ├── lib.rs              # Public API: model(), complete(), stream()
│   ├── types.rs             # Core types: Message, ContentBlock, Usage, Cost
│   ├── provider.rs          # Provider trait
│   ├── model.rs             # Model struct + registry
│   ├── stream.rs            # StreamEvent + streaming
│   ├── error.rs             # Unified error types
│   └── providers/
│       ├── mod.rs
│       ├── openai.rs
│       ├── anthropic.rs
│       ├── google.rs
│       ├── mistral.rs
│       ├── deepseek.rs
│       ├── groq.rs
│       ├── xai.rs
│       ├── ollama.rs
│       └── openai_compat.rs  # Generic OpenAI-compatible implementation
```

## Core Types

```rust
enum ContentBlock {
    Text(String),
    Thinking { content: String, signature: Option<String> },
    ToolCall { id: String, name: String, arguments: String },
    ToolResult { tool_call_id: String, content: String },
}

enum Role { System, User, Assistant, Tool }

struct Message {
    role: Role,
    content: Vec<ContentBlock>,
    timestamp: Option<i64>,
}

struct ChatRequest {
    messages: Vec<Message>,
    tools: Option<Vec<ToolDef>>,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    thinking: Option<ThinkingConfig>,
    stop: Option<Vec<String>>,
    abort_signal: Option<CancellationToken>,
}

struct ToolDef {
    name: String,
    description: String,
    parameters: serde_json::Value,  // JSON Schema
}

struct ThinkingConfig {
    enabled: bool,
    budget_tokens: Option<u32>,
}

struct ChatResponse {
    message: Message,
    usage: Usage,
    model: String,
}

struct Usage {
    input_tokens: u32,
    output_tokens: u32,
    thinking_tokens: Option<u32>,
    cache_read_tokens: Option<u32>,
    cache_write_tokens: Option<u32>,
    cost: Cost,
}

struct Cost {
    input: f64,
    output: f64,
    total: f64,
}

enum StreamEvent {
    TextDelta(String),
    ThinkingDelta(String),
    ToolCallDelta { id: String, name: Option<String>, arguments_delta: String },
    Usage(Usage),
    Done(ChatResponse),
    Error(Error),
}
```

## Provider Trait

```rust
#[async_trait]
trait Provider: Send + Sync {
    async fn complete(&self, req: &ChatRequest) -> Result<ChatResponse>;
    async fn stream(&self, req: &ChatRequest) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent>> + Send>>>;
    async fn list_models(&self) -> Result<Vec<ModelInfo>>;
    fn name(&self) -> &str;
}
```

## Public API

```rust
fn model(provider: &str, model_id: &str) -> Result<Model>;
async fn complete(model: &Model, req: ChatRequest) -> Result<ChatResponse>;
async fn stream(model: &Model, req: ChatRequest) -> Result<impl Stream<Item = Result<StreamEvent>>>;
async fn list_models(provider: &str) -> Result<Vec<ModelInfo>>;
fn register_provider(provider: impl Provider + 'static);
```

## Model Info

```rust
struct ModelInfo {
    id: String,
    provider: String,
    display_name: Option<String>,
    context_window: Option<u32>,
    max_output_tokens: Option<u32>,
    supports_tools: bool,
    supports_thinking: bool,
    input_price_per_million: Option<f64>,
    output_price_per_million: Option<f64>,
}
```

## Dependencies

- `tokio` — async runtime
- `reqwest` — HTTP client (JSON + stream)
- `serde` / `serde_json` — serialization
- `async-trait` — async trait support
- `futures` / `tokio-stream` — stream utilities
- `thiserror` — error types
- `tokio-util` — CancellationToken
- `tracing` — logging/tracing

## Feature Flags

Each provider is a feature flag, all enabled by default:
`openai`, `anthropic`, `google`, `mistral`, `deepseek`, `groq`, `xai`, `ollama`

## Reference

Architecture inspired by [pi-mono/pi-ai](https://github.com/badlogic/pi-mono) (TypeScript), adapted to idiomatic Rust patterns.
