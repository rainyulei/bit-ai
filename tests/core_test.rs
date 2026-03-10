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
