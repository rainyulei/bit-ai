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

#[derive(Debug, Clone)]
pub struct OpenAICompatConfig {
    pub name: String,
    pub base_url: String,
    pub api_key: String,
    pub default_model: Option<String>,
}

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
        format!(
            "{}/chat/completions",
            self.config.base_url.trim_end_matches('/')
        )
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
            body["tools"] = json!(tools
                .iter()
                .map(|t| {
                    json!({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters,
                        }
                    })
                })
                .collect::<Vec<_>>());
        }

        body
    }

    pub fn default_model(&self) -> &str {
        self.config.default_model.as_deref().unwrap_or("gpt-4o")
    }
}

fn convert_messages_to_openai(messages: &[Message]) -> Vec<serde_json::Value> {
    messages
        .iter()
        .map(|msg| {
            let role = match msg.role {
                Role::System => "system",
                Role::User => "user",
                Role::Assistant => "assistant",
                Role::Tool => "tool",
            };

            // Single text message
            if msg.content.len() == 1 {
                if let ContentBlock::Text { ref text } = msg.content[0] {
                    return json!({ "role": role, "content": text });
                }
                if let ContentBlock::ToolResult {
                    ref tool_call_id,
                    ref content,
                } = msg.content[0]
                {
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
                        ContentBlock::ToolCall {
                            id,
                            name,
                            arguments,
                        } => {
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
                .join("");

            json!({ "role": role, "content": text })
        })
        .collect()
}

fn parse_openai_response(body: &serde_json::Value) -> Result<(Message, Usage, String)> {
    let model = body["model"].as_str().unwrap_or("unknown").to_string();
    let choice = &body["choices"][0];
    let msg = &choice["message"];

    let mut content = Vec::new();

    if let Some(text) = msg["content"].as_str() {
        if !text.is_empty() {
            content.push(ContentBlock::Text {
                text: text.to_string(),
            });
        }
    }

    if let Some(tool_calls) = msg["tool_calls"].as_array() {
        for tc in tool_calls {
            content.push(ContentBlock::ToolCall {
                id: tc["id"].as_str().unwrap_or("").to_string(),
                name: tc["function"]["name"].as_str().unwrap_or("").to_string(),
                arguments: tc["function"]["arguments"]
                    .as_str()
                    .unwrap_or("{}")
                    .to_string(),
            });
        }
    }

    if content.is_empty() {
        content.push(ContentBlock::Text {
            text: String::new(),
        });
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

fn parse_openai_stream_chunk(
    chunk: &serde_json::Value,
    content_blocks: &mut Vec<ContentBlock>,
    usage: &mut Usage,
) -> Option<StreamEvent> {
    // Check for usage in chunk (some providers send it)
    if let Some(u) = chunk.get("usage") {
        if u.is_object() {
            usage.input_tokens = u["prompt_tokens"].as_u64().unwrap_or(0) as u32;
            usage.output_tokens = u["completion_tokens"].as_u64().unwrap_or(0) as u32;
            return Some(StreamEvent::Usage(usage.clone()));
        }
    }

    let choices = chunk["choices"].as_array()?;
    if choices.is_empty() {
        return None;
    }

    let choice = &choices[0];
    let delta = &choice["delta"];

    // Text delta
    if let Some(text) = delta["content"].as_str() {
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

    // Tool call delta
    if let Some(tool_calls) = delta["tool_calls"].as_array() {
        for tc in tool_calls {
            let id = tc["id"].as_str().unwrap_or("").to_string();
            let name = tc["function"]["name"].as_str().map(|s| s.to_string());
            let args = tc["function"]["arguments"]
                .as_str()
                .unwrap_or("")
                .to_string();

            if let Some(ref n) = name {
                content_blocks.push(ContentBlock::ToolCall {
                    id: id.clone(),
                    name: n.clone(),
                    arguments: args.clone(),
                });
            } else if !args.is_empty() {
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

    None
}

#[async_trait]
impl Provider for OpenAICompatProvider {
    async fn complete(&self, req: &ChatRequest) -> Result<ChatResponse> {
        let model_id = self.default_model();
        let mut body = self.build_request_body(req, model_id);
        body["stream"] = json!(false);

        let resp = self
            .client
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
        let (message, usage, model) = parse_openai_response(&resp_body)?;

        Ok(ChatResponse {
            message,
            usage,
            model,
        })
    }

    async fn stream(&self, req: &ChatRequest) -> Result<BoxStream<'_, Result<StreamEvent>>> {
        let model_id = self.default_model();
        let mut body = self.build_request_body(req, model_id);
        body["stream"] = json!(true);

        let resp = self
            .client
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
                            if sse_event.data.is_some() {
                                if sse_event.is_done() {
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
                                        model: String::new(),
                                    };

                                    return Some((
                                        Ok(StreamEvent::Done(resp)),
                                        (
                                            byte_stream,
                                            buffer,
                                            sse_event,
                                            content_blocks,
                                            usage,
                                        ),
                                    ));
                                }

                                if let Some(ref data) = sse_event.data {
                                    if let Ok(chunk) =
                                        serde_json::from_str::<serde_json::Value>(data)
                                    {
                                        let evt = parse_openai_stream_chunk(
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
                                        continue;
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
                                (byte_stream, buffer, sse_event, content_blocks, usage),
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
        let resp = self
            .client
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
        let empty = Vec::new();
        let models = body["data"]
            .as_array()
            .unwrap_or(&empty)
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
