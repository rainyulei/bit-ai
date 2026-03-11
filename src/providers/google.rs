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

pub struct GoogleProvider {
    api_key: String,
    client: Client,
}

impl GoogleProvider {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            client: Client::new(),
        }
    }

    pub fn from_env() -> Result<Self> {
        let key = std::env::var("GOOGLE_API_KEY")
            .map_err(|_| BitAiError::Config("GOOGLE_API_KEY not set".to_string()))?;
        Ok(Self::new(key))
    }

    fn generate_url(&self, model: &str) -> String {
        format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
            model, self.api_key
        )
    }

    fn stream_url(&self, model: &str) -> String {
        format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:streamGenerateContent?key={}&alt=sse",
            model, self.api_key
        )
    }

    fn build_request_body(&self, req: &ChatRequest) -> serde_json::Value {
        let mut body = json!({});

        // Extract system instruction and convert messages
        let (system, contents) = convert_messages_to_google(&req.messages);

        if let Some(sys) = system {
            body["system_instruction"] = json!({
                "parts": [{ "text": sys }]
            });
        }

        body["contents"] = json!(contents);

        // Generation config
        let mut gen_config = json!({});
        if let Some(temp) = req.temperature {
            gen_config["temperature"] = json!(temp);
        }
        if let Some(max) = req.max_tokens {
            gen_config["maxOutputTokens"] = json!(max);
        }
        if let Some(ref stop) = req.stop {
            gen_config["stopSequences"] = json!(stop);
        }
        if gen_config.as_object().map_or(false, |o| !o.is_empty()) {
            body["generationConfig"] = gen_config;
        }

        // Tools
        if let Some(ref tools) = req.tools {
            let function_declarations: Vec<serde_json::Value> = tools
                .iter()
                .map(|t| {
                    json!({
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    })
                })
                .collect();
            body["tools"] = json!([{
                "function_declarations": function_declarations,
            }]);
        }

        body
    }

    fn default_model(&self) -> &str {
        "gemini-2.5-flash"
    }

    fn known_models() -> Vec<ModelInfo> {
        vec![
            ModelInfo {
                id: "gemini-2.5-pro".to_string(),
                provider: "google".to_string(),
                display_name: Some("Gemini 2.5 Pro".to_string()),
                context_window: Some(1_048_576),
                max_output_tokens: Some(65_536),
                supports_tools: true,
                supports_thinking: true,
                input_price_per_million: Some(1.25),
                output_price_per_million: Some(10.00),
            },
            ModelInfo {
                id: "gemini-2.5-flash".to_string(),
                provider: "google".to_string(),
                display_name: Some("Gemini 2.5 Flash".to_string()),
                context_window: Some(1_048_576),
                max_output_tokens: Some(65_536),
                supports_tools: true,
                supports_thinking: true,
                input_price_per_million: Some(0.15),
                output_price_per_million: Some(0.60),
            },
            ModelInfo {
                id: "gemini-2.0-flash".to_string(),
                provider: "google".to_string(),
                display_name: Some("Gemini 2.0 Flash".to_string()),
                context_window: Some(1_048_576),
                max_output_tokens: Some(8192),
                supports_tools: true,
                supports_thinking: false,
                input_price_per_million: Some(0.10),
                output_price_per_million: Some(0.40),
            },
        ]
    }
}

fn convert_messages_to_google(
    messages: &[Message],
) -> (Option<String>, Vec<serde_json::Value>) {
    let mut system = None;
    let mut contents = Vec::new();

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
                let parts: Vec<serde_json::Value> = msg
                    .content
                    .iter()
                    .filter_map(|b| match b {
                        ContentBlock::Text { text } => Some(json!({ "text": text })),
                        _ => None,
                    })
                    .collect();
                contents.push(json!({ "role": "user", "parts": parts }));
            }
            Role::Assistant => {
                let parts: Vec<serde_json::Value> = msg
                    .content
                    .iter()
                    .filter_map(|b| match b {
                        ContentBlock::Text { text } => Some(json!({ "text": text })),
                        ContentBlock::ToolCall {
                            name, arguments, ..
                        } => {
                            let args: serde_json::Value =
                                serde_json::from_str(arguments).unwrap_or(json!({}));
                            Some(json!({
                                "functionCall": {
                                    "name": name,
                                    "args": args,
                                }
                            }))
                        }
                        _ => None,
                    })
                    .collect();
                contents.push(json!({ "role": "model", "parts": parts }));
            }
            Role::Tool => {
                let parts: Vec<serde_json::Value> = msg
                    .content
                    .iter()
                    .filter_map(|b| {
                        if let ContentBlock::ToolResult {
                            tool_call_id,
                            content,
                        } = b
                        {
                            Some(json!({
                                "functionResponse": {
                                    "name": tool_call_id,
                                    "response": {
                                        "content": content,
                                    }
                                }
                            }))
                        } else {
                            None
                        }
                    })
                    .collect();
                contents.push(json!({ "role": "user", "parts": parts }));
            }
        }
    }

    (system, contents)
}

fn parse_google_response(body: &serde_json::Value) -> Result<(Message, Usage)> {
    let mut content = Vec::new();

    if let Some(candidates) = body["candidates"].as_array() {
        if let Some(candidate) = candidates.first() {
            if let Some(parts) = candidate["content"]["parts"].as_array() {
                for part in parts {
                    if let Some(text) = part["text"].as_str() {
                        content.push(ContentBlock::Text {
                            text: text.to_string(),
                        });
                    }
                    if let Some(fc) = part.get("functionCall") {
                        content.push(ContentBlock::ToolCall {
                            id: fc["name"].as_str().unwrap_or("").to_string(),
                            name: fc["name"].as_str().unwrap_or("").to_string(),
                            arguments: fc["args"].to_string(),
                        });
                    }
                }
            }
        }
    }

    if content.is_empty() {
        content.push(ContentBlock::Text {
            text: String::new(),
        });
    }

    let usage_meta = &body["usageMetadata"];
    let usage = Usage {
        input_tokens: usage_meta["promptTokenCount"].as_u64().unwrap_or(0) as u32,
        output_tokens: usage_meta["candidatesTokenCount"].as_u64().unwrap_or(0) as u32,
        ..Default::default()
    };

    let message = Message {
        role: Role::Assistant,
        content,
        timestamp: None,
    };

    Ok((message, usage))
}

fn parse_google_stream_chunk(
    chunk: &serde_json::Value,
    content_blocks: &mut Vec<ContentBlock>,
    usage: &mut Usage,
) -> Option<StreamEvent> {
    // Update usage if present
    if let Some(usage_meta) = chunk.get("usageMetadata") {
        usage.input_tokens = usage_meta["promptTokenCount"].as_u64().unwrap_or(0) as u32;
        usage.output_tokens = usage_meta["candidatesTokenCount"].as_u64().unwrap_or(0) as u32;
    }

    if let Some(candidates) = chunk["candidates"].as_array() {
        if let Some(candidate) = candidates.first() {
            if let Some(parts) = candidate["content"]["parts"].as_array() {
                for part in parts {
                    if let Some(text) = part["text"].as_str() {
                        if !text.is_empty() {
                            let found = content_blocks
                                .iter_mut()
                                .find(|b| matches!(b, ContentBlock::Text { .. }));
                            if let Some(ContentBlock::Text { text: t }) = found {
                                t.push_str(text);
                            } else {
                                content_blocks.push(ContentBlock::Text {
                                    text: text.to_string(),
                                });
                            }
                            return Some(StreamEvent::TextDelta(text.to_string()));
                        }
                    }
                    if let Some(fc) = part.get("functionCall") {
                        let name = fc["name"].as_str().unwrap_or("").to_string();
                        let args = fc["args"].to_string();
                        content_blocks.push(ContentBlock::ToolCall {
                            id: name.clone(),
                            name: name.clone(),
                            arguments: args.clone(),
                        });
                        return Some(StreamEvent::ToolCallDelta {
                            id: name.clone(),
                            name: Some(name),
                            arguments_delta: args,
                        });
                    }
                }
            }
        }
    }

    None
}

#[async_trait]
impl Provider for GoogleProvider {
    async fn complete(&self, req: &ChatRequest) -> Result<ChatResponse> {
        let model_id = self.default_model();
        let body = self.build_request_body(req);

        let resp = self
            .client
            .post(&self.generate_url(model_id))
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
        let (message, usage) = parse_google_response(&resp_body)?;

        Ok(ChatResponse {
            message,
            usage,
            model: model_id.to_string(),
        })
    }

    async fn stream(&self, req: &ChatRequest) -> Result<BoxStream<'_, Result<StreamEvent>>> {
        let model_id = self.default_model();
        let body = self.build_request_body(req);

        let resp = self
            .client
            .post(&self.stream_url(model_id))
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
                model_id.to_string(),
            ),
            |(mut byte_stream, mut buffer, mut sse_event, mut content_blocks, mut usage, model)| async move {
                loop {
                    if let Some(newline_pos) = buffer.find('\n') {
                        let line = buffer[..newline_pos].trim_end_matches('\r').to_string();
                        buffer = buffer[newline_pos + 1..].to_string();

                        if line.is_empty() {
                            if let Some(ref data) = sse_event.data {
                                if let Ok(chunk) =
                                    serde_json::from_str::<serde_json::Value>(data)
                                {
                                    let evt = parse_google_stream_chunk(
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
                                                model,
                                            ),
                                        ));
                                    }
                                    continue;
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
                                (byte_stream, buffer, sse_event, content_blocks, usage, model),
                            ));
                        }
                        None => {
                            // Stream ended — emit Done event
                            if !content_blocks.is_empty() || usage.input_tokens > 0 {
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
                                let resp = ChatResponse {
                                    message,
                                    usage: usage.clone(),
                                    model: model.clone(),
                                };
                                // Clear content_blocks to prevent re-emitting
                                content_blocks.clear();
                                return Some((
                                    Ok(StreamEvent::Done(resp)),
                                    (byte_stream, buffer, sse_event, content_blocks, usage, model),
                                ));
                            }
                            return None;
                        }
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
        "google"
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
        let (system, contents) = convert_messages_to_google(&messages);
        assert_eq!(system, Some("You are helpful.".to_string()));
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0]["role"], "user");
    }

    #[test]
    fn test_convert_user_message() {
        let messages = vec![Message::user("Hello")];
        let (system, contents) = convert_messages_to_google(&messages);
        assert!(system.is_none());
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0]["role"], "user");
        let parts = contents[0]["parts"].as_array().unwrap();
        assert_eq!(parts[0]["text"], "Hello");
    }

    #[test]
    fn test_convert_assistant_message() {
        let messages = vec![Message::assistant("Hi there")];
        let (_system, contents) = convert_messages_to_google(&messages);
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0]["role"], "model");
        let parts = contents[0]["parts"].as_array().unwrap();
        assert_eq!(parts[0]["text"], "Hi there");
    }

    #[test]
    fn test_convert_tool_call_message() {
        let messages = vec![Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolCall {
                id: "tc_1".to_string(),
                name: "get_weather".to_string(),
                arguments: r#"{"city":"SF"}"#.to_string(),
            }],
            timestamp: None,
        }];
        let (_system, contents) = convert_messages_to_google(&messages);
        assert_eq!(contents[0]["role"], "model");
        let parts = contents[0]["parts"].as_array().unwrap();
        assert_eq!(parts[0]["functionCall"]["name"], "get_weather");
        assert_eq!(parts[0]["functionCall"]["args"]["city"], "SF");
    }

    #[test]
    fn test_parse_google_response_text() {
        let body = json!({
            "candidates": [{
                "content": {
                    "parts": [{ "text": "Hello!" }]
                }
            }],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5
            }
        });
        let (msg, usage) = parse_google_response(&body).unwrap();
        assert_eq!(msg.role, Role::Assistant);
        assert_eq!(msg.content.len(), 1);
        if let ContentBlock::Text { ref text } = msg.content[0] {
            assert_eq!(text, "Hello!");
        } else {
            panic!("Expected Text block");
        }
        assert_eq!(usage.input_tokens, 10);
        assert_eq!(usage.output_tokens, 5);
    }

    #[test]
    fn test_parse_google_response_function_call() {
        let body = json!({
            "candidates": [{
                "content": {
                    "parts": [{
                        "functionCall": {
                            "name": "get_weather",
                            "args": { "city": "SF" }
                        }
                    }]
                }
            }],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5
            }
        });
        let (msg, _usage) = parse_google_response(&body).unwrap();
        assert_eq!(msg.content.len(), 1);
        if let ContentBlock::ToolCall {
            ref name,
            ref arguments,
            ..
        } = msg.content[0]
        {
            assert_eq!(name, "get_weather");
            let parsed: serde_json::Value = serde_json::from_str(arguments).unwrap();
            assert_eq!(parsed["city"], "SF");
        } else {
            panic!("Expected ToolCall block");
        }
    }

    #[test]
    fn test_parse_empty_response() {
        let body = json!({
            "candidates": [{
                "content": { "parts": [] }
            }],
            "usageMetadata": {}
        });
        let (msg, _usage) = parse_google_response(&body).unwrap();
        assert_eq!(msg.content.len(), 1);
        if let ContentBlock::Text { ref text } = msg.content[0] {
            assert!(text.is_empty());
        } else {
            panic!("Expected empty Text block");
        }
    }

    #[test]
    fn test_build_request_body_basic() {
        let provider = GoogleProvider::new("test-key");
        let req = ChatRequest::new(vec![Message::user("Hello")]);
        let body = provider.build_request_body(&req);
        assert!(body.get("contents").is_some());
        assert!(body.get("system_instruction").is_none());
    }

    #[test]
    fn test_build_request_body_with_system() {
        let provider = GoogleProvider::new("test-key");
        let req = ChatRequest::new(vec![
            Message::system("Be helpful"),
            Message::user("Hello"),
        ]);
        let body = provider.build_request_body(&req);
        assert_eq!(body["system_instruction"]["parts"][0]["text"], "Be helpful");
    }

    #[test]
    fn test_build_request_body_with_tools() {
        let provider = GoogleProvider::new("test-key");
        let req = ChatRequest::new(vec![Message::user("Hello")]).with_tools(vec![ToolDef {
            name: "get_weather".to_string(),
            description: "Get weather".to_string(),
            parameters: json!({"type": "object", "properties": {"city": {"type": "string"}}}),
        }]);
        let body = provider.build_request_body(&req);
        let tools = body["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        let decls = tools[0]["function_declarations"].as_array().unwrap();
        assert_eq!(decls[0]["name"], "get_weather");
    }

    #[test]
    fn test_known_models() {
        let models = GoogleProvider::known_models();
        assert_eq!(models.len(), 3);
        assert!(models.iter().all(|m| m.provider == "google"));
        assert!(models.iter().all(|m| m.supports_tools));
    }

    #[test]
    fn test_provider_name() {
        let provider = GoogleProvider::new("test-key");
        assert_eq!(provider.name(), "google");
    }

    #[test]
    fn test_stream_chunk_text() {
        let chunk = json!({
            "candidates": [{
                "content": {
                    "parts": [{ "text": "Hello" }]
                }
            }]
        });
        let mut blocks = Vec::new();
        let mut usage = Usage::default();
        let evt = parse_google_stream_chunk(&chunk, &mut blocks, &mut usage);
        assert!(matches!(evt, Some(StreamEvent::TextDelta(ref t)) if t == "Hello"));
        assert_eq!(blocks.len(), 1);
    }

    #[test]
    fn test_stream_chunk_function_call() {
        let chunk = json!({
            "candidates": [{
                "content": {
                    "parts": [{
                        "functionCall": {
                            "name": "get_weather",
                            "args": { "city": "SF" }
                        }
                    }]
                }
            }]
        });
        let mut blocks = Vec::new();
        let mut usage = Usage::default();
        let evt = parse_google_stream_chunk(&chunk, &mut blocks, &mut usage);
        assert!(matches!(evt, Some(StreamEvent::ToolCallDelta { .. })));
        assert_eq!(blocks.len(), 1);
    }
}
