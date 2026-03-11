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
            body["tools"] = json!(tools
                .iter()
                .map(|t| {
                    json!({
                        "name": t.name,
                        "description": t.description,
                        "input_schema": t.parameters,
                    })
                })
                .collect::<Vec<_>>());
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

fn convert_messages_to_anthropic(
    messages: &[Message],
) -> (Option<String>, Vec<serde_json::Value>) {
    let mut system = None;
    let mut result = Vec::new();

    for msg in messages {
        match msg.role {
            Role::System => {
                let text: String = msg
                    .content
                    .iter()
                    .filter_map(|b| {
                        if let ContentBlock::Text { text } = b {
                            Some(text.as_str())
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                system = Some(text);
            }
            Role::User => {
                let content: Vec<serde_json::Value> = msg
                    .content
                    .iter()
                    .map(|b| match b {
                        ContentBlock::Text { text } => json!({ "type": "text", "text": text }),
                        ContentBlock::ToolResult {
                            tool_call_id,
                            content,
                        } => json!({
                            "type": "tool_result",
                            "tool_use_id": tool_call_id,
                            "content": content,
                        }),
                        _ => json!({ "type": "text", "text": "" }),
                    })
                    .collect();
                result.push(json!({ "role": "user", "content": content }));
            }
            Role::Assistant => {
                let content: Vec<serde_json::Value> = msg
                    .content
                    .iter()
                    .filter_map(|b| match b {
                        ContentBlock::Text { text } => {
                            Some(json!({ "type": "text", "text": text }))
                        }
                        ContentBlock::Thinking {
                            content,
                            signature,
                        } => Some(json!({
                            "type": "thinking",
                            "thinking": content,
                            "signature": signature,
                        })),
                        ContentBlock::ToolCall {
                            id,
                            name,
                            arguments,
                        } => {
                            let input: serde_json::Value =
                                serde_json::from_str(arguments).unwrap_or(json!({}));
                            Some(json!({
                                "type": "tool_use",
                                "id": id,
                                "name": name,
                                "input": input,
                            }))
                        }
                        _ => None,
                    })
                    .collect();
                result.push(json!({ "role": "assistant", "content": content }));
            }
            Role::Tool => {
                let content: Vec<serde_json::Value> = msg
                    .content
                    .iter()
                    .filter_map(|b| {
                        if let ContentBlock::ToolResult {
                            tool_call_id,
                            content,
                        } = b
                        {
                            Some(json!({
                                "type": "tool_result",
                                "tool_use_id": tool_call_id,
                                "content": content,
                            }))
                        } else {
                            None
                        }
                    })
                    .collect();
                result.push(json!({ "role": "user", "content": content }));
            }
        }
    }

    (system, result)
}

fn parse_anthropic_response(body: &serde_json::Value) -> Message {
    let mut content = Vec::new();

    if let Some(blocks) = body["content"].as_array() {
        for block in blocks {
            match block["type"].as_str() {
                Some("text") => {
                    if let Some(text) = block["text"].as_str() {
                        content.push(ContentBlock::Text {
                            text: text.to_string(),
                        });
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

    if content.is_empty() {
        content.push(ContentBlock::Text {
            text: String::new(),
        });
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
                    content_blocks.push(ContentBlock::Text {
                        text: String::new(),
                    });
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
                    if let Some(ContentBlock::Text { text: t }) = content_blocks.last_mut()
                    {
                        t.push_str(&text);
                    }
                    Some(StreamEvent::TextDelta(text))
                }
                Some("thinking_delta") => {
                    let thinking = delta["thinking"].as_str().unwrap_or("").to_string();
                    if let Some(ContentBlock::Thinking {
                        content, ..
                    }) = content_blocks.last_mut()
                    {
                        content.push_str(&thinking);
                    }
                    Some(StreamEvent::ThinkingDelta(thinking))
                }
                Some("input_json_delta") => {
                    let partial = delta["partial_json"].as_str().unwrap_or("").to_string();
                    let (id, name) = if let Some(ContentBlock::ToolCall {
                        id,
                        name,
                        arguments,
                    }) = content_blocks.last_mut()
                    {
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
                    if let Some(ContentBlock::Thinking {
                        signature, ..
                    }) = content_blocks.last_mut()
                    {
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
                content: if content_blocks.is_empty() {
                    vec![ContentBlock::Text {
                        text: String::new(),
                    }]
                } else {
                    content_blocks.clone()
                },
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

#[async_trait]
impl Provider for AnthropicProvider {
    async fn complete(&self, req: &ChatRequest) -> Result<ChatResponse> {
        let model_id = "claude-sonnet-4-20250514";
        let body = self.build_request_body(req, model_id);

        let resp = self
            .client
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
        let model = resp_body["model"]
            .as_str()
            .unwrap_or(model_id)
            .to_string();

        Ok(ChatResponse {
            message,
            usage,
            model,
        })
    }

    async fn stream(&self, req: &ChatRequest) -> Result<BoxStream<'_, Result<StreamEvent>>> {
        let model_id = "claude-sonnet-4-20250514";
        let mut body = self.build_request_body(req, model_id);
        body["stream"] = json!(true);

        let resp = self
            .client
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
            (
                byte_stream,
                String::new(),
                SseEvent::default(),
                Vec::<ContentBlock>::new(),
                Usage::default(),
            ),
            |(mut byte_stream, mut buffer, mut sse_event, mut content_blocks, mut usage)| async move {
                loop {
                    if let Some(newline_pos) = buffer.find('\n') {
                        let line = buffer[..newline_pos].trim_end_matches('\r').to_string();
                        buffer = buffer[newline_pos + 1..].to_string();

                        if line.is_empty() {
                            if let Some(ref data) = sse_event.data {
                                if let Ok(chunk) =
                                    serde_json::from_str::<serde_json::Value>(data)
                                {
                                    let event_type =
                                        sse_event.event.as_deref().unwrap_or("");
                                    let evt = parse_anthropic_stream_event(
                                        event_type,
                                        &chunk,
                                        &mut content_blocks,
                                        &mut usage,
                                    );
                                    sse_event.reset();
                                    if let Some(event) = evt {
                                        return Some((
                                            Ok(event),
                                            (
                                                byte_stream,
                                                buffer,
                                                sse_event,
                                                content_blocks,
                                                usage,
                                            ),
                                        ));
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
                            return Some((
                                Err(BitAiError::Http(e)),
                                (
                                    byte_stream,
                                    buffer,
                                    sse_event,
                                    content_blocks,
                                    usage,
                                ),
                            ));
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_system_message() {
        let messages = vec![
            Message::system("You are helpful."),
            Message::user("Hello"),
        ];
        let (system, converted) = convert_messages_to_anthropic(&messages);
        assert_eq!(system, Some("You are helpful.".to_string()));
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0]["role"], "user");
    }

    #[test]
    fn test_convert_tool_call_message() {
        let messages = vec![Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Text {
                    text: "Let me check.".to_string(),
                },
                ContentBlock::ToolCall {
                    id: "tc_1".to_string(),
                    name: "get_weather".to_string(),
                    arguments: r#"{"city":"SF"}"#.to_string(),
                },
            ],
            timestamp: None,
        }];
        let (system, converted) = convert_messages_to_anthropic(&messages);
        assert!(system.is_none());
        assert_eq!(converted.len(), 1);
        let content = converted[0]["content"].as_array().unwrap();
        assert_eq!(content.len(), 2);
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[1]["type"], "tool_use");
        assert_eq!(content[1]["id"], "tc_1");
        assert_eq!(content[1]["name"], "get_weather");
        assert_eq!(content[1]["input"]["city"], "SF");
    }

    #[test]
    fn test_convert_tool_result_message() {
        let messages = vec![Message {
            role: Role::Tool,
            content: vec![ContentBlock::ToolResult {
                tool_call_id: "tc_1".to_string(),
                content: "72F and sunny".to_string(),
            }],
            timestamp: None,
        }];
        let (_system, converted) = convert_messages_to_anthropic(&messages);
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0]["role"], "user");
        let content = converted[0]["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], "tool_result");
        assert_eq!(content[0]["tool_use_id"], "tc_1");
    }

    #[test]
    fn test_parse_anthropic_response_text() {
        let body = json!({
            "content": [
                { "type": "text", "text": "Hello!" }
            ],
            "usage": { "input_tokens": 10, "output_tokens": 5 }
        });
        let msg = parse_anthropic_response(&body);
        assert_eq!(msg.role, Role::Assistant);
        assert_eq!(msg.content.len(), 1);
        if let ContentBlock::Text { ref text } = msg.content[0] {
            assert_eq!(text, "Hello!");
        } else {
            panic!("Expected Text block");
        }
    }

    #[test]
    fn test_parse_anthropic_response_tool_use() {
        let body = json!({
            "content": [
                { "type": "text", "text": "Let me check." },
                {
                    "type": "tool_use",
                    "id": "tu_1",
                    "name": "get_weather",
                    "input": { "city": "SF" }
                }
            ]
        });
        let msg = parse_anthropic_response(&body);
        assert_eq!(msg.content.len(), 2);
        if let ContentBlock::ToolCall {
            ref id,
            ref name,
            ref arguments,
        } = msg.content[1]
        {
            assert_eq!(id, "tu_1");
            assert_eq!(name, "get_weather");
            let parsed: serde_json::Value = serde_json::from_str(arguments).unwrap();
            assert_eq!(parsed["city"], "SF");
        } else {
            panic!("Expected ToolCall block");
        }
    }

    #[test]
    fn test_parse_anthropic_response_thinking() {
        let body = json!({
            "content": [
                {
                    "type": "thinking",
                    "thinking": "Let me reason...",
                    "signature": "sig123"
                },
                { "type": "text", "text": "The answer is 42." }
            ]
        });
        let msg = parse_anthropic_response(&body);
        assert_eq!(msg.content.len(), 2);
        if let ContentBlock::Thinking {
            ref content,
            ref signature,
        } = msg.content[0]
        {
            assert_eq!(content, "Let me reason...");
            assert_eq!(signature.as_deref(), Some("sig123"));
        } else {
            panic!("Expected Thinking block");
        }
    }

    #[test]
    fn test_parse_anthropic_usage() {
        let body = json!({
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_read_input_tokens": 20,
                "cache_creation_input_tokens": 10
            }
        });
        let usage = parse_anthropic_usage(&body);
        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.output_tokens, 50);
        assert_eq!(usage.cache_read_tokens, Some(20));
        assert_eq!(usage.cache_write_tokens, Some(10));
    }

    #[test]
    fn test_parse_empty_response() {
        let body = json!({ "content": [] });
        let msg = parse_anthropic_response(&body);
        assert_eq!(msg.content.len(), 1);
        if let ContentBlock::Text { ref text } = msg.content[0] {
            assert!(text.is_empty());
        } else {
            panic!("Expected empty Text block");
        }
    }

    #[test]
    fn test_stream_event_message_start() {
        let data = json!({
            "message": {
                "usage": { "input_tokens": 42 }
            }
        });
        let mut blocks = Vec::new();
        let mut usage = Usage::default();
        let evt = parse_anthropic_stream_event("message_start", &data, &mut blocks, &mut usage);
        assert!(evt.is_none());
        assert_eq!(usage.input_tokens, 42);
    }

    #[test]
    fn test_stream_event_text_delta() {
        let data = json!({
            "delta": { "type": "text_delta", "text": "Hello" }
        });
        let mut blocks = vec![ContentBlock::Text {
            text: String::new(),
        }];
        let mut usage = Usage::default();
        let evt =
            parse_anthropic_stream_event("content_block_delta", &data, &mut blocks, &mut usage);
        assert!(matches!(evt, Some(StreamEvent::TextDelta(ref t)) if t == "Hello"));
        if let ContentBlock::Text { ref text } = blocks[0] {
            assert_eq!(text, "Hello");
        }
    }

    #[test]
    fn test_stream_event_tool_call_delta() {
        let data = json!({
            "delta": { "type": "input_json_delta", "partial_json": "{\"city\":" }
        });
        let mut blocks = vec![ContentBlock::ToolCall {
            id: "tc_1".to_string(),
            name: "get_weather".to_string(),
            arguments: String::new(),
        }];
        let mut usage = Usage::default();
        let evt =
            parse_anthropic_stream_event("content_block_delta", &data, &mut blocks, &mut usage);
        assert!(matches!(evt, Some(StreamEvent::ToolCallDelta { .. })));
        if let ContentBlock::ToolCall { ref arguments, .. } = blocks[0] {
            assert_eq!(arguments, "{\"city\":");
        }
    }

    #[test]
    fn test_stream_event_message_stop() {
        let data = json!({});
        let mut blocks = vec![ContentBlock::Text {
            text: "Done".to_string(),
        }];
        let mut usage = Usage::default();
        let evt = parse_anthropic_stream_event("message_stop", &data, &mut blocks, &mut usage);
        assert!(matches!(evt, Some(StreamEvent::Done(_))));
    }

    #[test]
    fn test_build_request_body_basic() {
        let provider = AnthropicProvider::new("test-key");
        let req = ChatRequest::new(vec![Message::user("Hello")]);
        let body = provider.build_request_body(&req, "claude-sonnet-4-20250514");
        assert_eq!(body["model"], "claude-sonnet-4-20250514");
        assert_eq!(body["max_tokens"], 4096);
        assert!(body.get("system").is_none());
    }

    #[test]
    fn test_build_request_body_with_thinking() {
        let provider = AnthropicProvider::new("test-key");
        let req = ChatRequest::new(vec![Message::user("Hello")]).with_thinking(ThinkingConfig {
            enabled: true,
            budget_tokens: Some(5000),
        });
        let body = provider.build_request_body(&req, "claude-sonnet-4-20250514");
        assert_eq!(body["thinking"]["type"], "enabled");
        assert_eq!(body["thinking"]["budget_tokens"], 5000);
    }

    #[test]
    fn test_build_request_body_with_tools() {
        let provider = AnthropicProvider::new("test-key");
        let req = ChatRequest::new(vec![Message::user("Hello")]).with_tools(vec![ToolDef {
            name: "get_weather".to_string(),
            description: "Get weather".to_string(),
            parameters: json!({"type": "object", "properties": {"city": {"type": "string"}}}),
        }]);
        let body = provider.build_request_body(&req, "claude-sonnet-4-20250514");
        let tools = body["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["name"], "get_weather");
        assert!(tools[0].get("input_schema").is_some());
    }

    #[test]
    fn test_known_models() {
        let models = AnthropicProvider::known_models();
        assert_eq!(models.len(), 3);
        assert!(models.iter().all(|m| m.provider == "anthropic"));
        assert!(models.iter().all(|m| m.supports_tools));
        assert!(models.iter().all(|m| m.supports_thinking));
    }

    #[test]
    fn test_provider_name() {
        let provider = AnthropicProvider::new("test-key");
        assert_eq!(provider.name(), "anthropic");
    }
}
