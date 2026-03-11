# bit-ai

Unified multi-provider LLM API for Rust. One interface, many providers.

## Supported Providers

| Provider | API Key Env Var | Protocol |
|----------|----------------|----------|
| OpenAI | `OPENAI_API_KEY` | OpenAI-compatible |
| Anthropic | `ANTHROPIC_API_KEY` | Messages API |
| Google Gemini | `GOOGLE_API_KEY` | Gemini API |
| DeepSeek | `DEEPSEEK_API_KEY` | OpenAI-compatible |
| Groq | `GROQ_API_KEY` | OpenAI-compatible |
| xAI (Grok) | `XAI_API_KEY` | OpenAI-compatible |
| Mistral | `MISTRAL_API_KEY` | OpenAI-compatible |
| Ollama | _(none, local)_ | OpenAI-compatible |

## Quick Start

Add to `Cargo.toml`:

```toml
[dependencies]
bit-ai = { git = "https://github.com/rainyulei/bit-ai.git" }
tokio = { version = "1", features = ["full"] }
futures = "0.3"
```

### Non-streaming

```rust
use bit_ai::*;

#[tokio::main]
async fn main() -> bit_ai::Result<()> {
    bit_ai::init(); // auto-register providers from env vars

    let model = bit_ai::model("openai", "gpt-4o")?;
    let req = ChatRequest::new(vec![Message::user("Hello!")]);
    let resp = bit_ai::complete(&model, req).await?;

    println!("{:?}", resp.message.content);
    println!("Tokens: {} in / {} out", resp.usage.input_tokens, resp.usage.output_tokens);
    Ok(())
}
```

### Streaming

```rust
use bit_ai::*;
use futures::StreamExt;

#[tokio::main]
async fn main() -> bit_ai::Result<()> {
    bit_ai::init();

    let model = bit_ai::model("anthropic", "claude-sonnet-4-20250514")?;
    let req = ChatRequest::new(vec![Message::user("Explain Rust in 3 sentences.")]);
    let mut stream = bit_ai::stream(&model, req).await?;

    while let Some(event) = stream.next().await {
        match event? {
            StreamEvent::TextDelta(text) => print!("{text}"),
            StreamEvent::Done(resp) => {
                println!("\n\nTokens: {} in / {} out", resp.usage.input_tokens, resp.usage.output_tokens);
            }
            _ => {}
        }
    }
    Ok(())
}
```

### Tool Calling

```rust
use bit_ai::*;
use serde_json::json;

let req = ChatRequest::new(vec![Message::user("What's the weather in Tokyo?")])
    .with_tools(vec![ToolDef {
        name: "get_weather".to_string(),
        description: "Get current weather for a city".to_string(),
        parameters: json!({
            "type": "object",
            "properties": {
                "city": { "type": "string" }
            },
            "required": ["city"]
        }),
    }]);
```

### Thinking / Reasoning

```rust
let req = ChatRequest::new(vec![Message::user("Solve this step by step: ...")])
    .with_thinking(ThinkingConfig {
        enabled: true,
        budget_tokens: Some(10000),
    });
```

## API

```rust
// Initialize (auto-register from env vars)
bit_ai::init();

// Get a model handle
let model = bit_ai::model("provider", "model-id")?;

// Non-streaming completion
let resp = bit_ai::complete(&model, req).await?;

// Streaming completion
let stream = bit_ai::stream(&model, req).await?;

// List models for a provider
let models = bit_ai::list_models("openai").await?;

// Register a custom provider
bit_ai::register_provider(Arc::new(my_provider));
```

## Feature Flags

Each provider can be individually enabled/disabled. All enabled by default.

```toml
# Only include OpenAI and Anthropic
bit-ai = { git = "...", default-features = false, features = ["openai", "anthropic"] }
```

Available features: `openai`, `anthropic`, `google`, `mistral`, `deepseek`, `groq`, `xai`, `ollama`

## Custom Provider

Implement the `Provider` trait to add your own:

```rust
use bit_ai::{Provider, ChatRequest, ChatResponse, StreamEvent, ModelInfo, Result};
use async_trait::async_trait;
use futures::stream::BoxStream;

struct MyProvider;

#[async_trait]
impl Provider for MyProvider {
    async fn complete(&self, req: &ChatRequest) -> Result<ChatResponse> { todo!() }
    async fn stream(&self, req: &ChatRequest) -> Result<BoxStream<'_, Result<StreamEvent>>> { todo!() }
    async fn list_models(&self) -> Result<Vec<ModelInfo>> { todo!() }
    fn name(&self) -> &str { "my-provider" }
}
```

## License

MIT
