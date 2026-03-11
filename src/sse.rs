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
