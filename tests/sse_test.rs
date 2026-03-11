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
