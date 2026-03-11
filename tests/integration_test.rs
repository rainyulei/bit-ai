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
    println!("OpenAI response: {:?}", resp.message.content);
    println!("Usage: {:?}", resp.usage);
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
            StreamEvent::TextDelta(t) => {
                got_text = true;
                print!("{}", t);
            }
            StreamEvent::Done(resp) => {
                println!("\nDone. Usage: {:?}", resp.usage);
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
    println!("Anthropic response: {:?}", resp.message.content);
    println!("Usage: {:?}", resp.usage);
}

#[tokio::test]
#[ignore] // Requires ANTHROPIC_API_KEY
async fn test_anthropic_stream() {
    bit_ai::init();
    let model = bit_ai::model("anthropic", "claude-haiku-4-20250514").unwrap();
    let req = ChatRequest::new(vec![Message::user("Say 'hello' and nothing else.")]);
    let mut stream = bit_ai::stream(&model, req).await.unwrap();

    let mut got_text = false;
    while let Some(event) = stream.next().await {
        match event.unwrap() {
            StreamEvent::TextDelta(t) => {
                got_text = true;
                print!("{}", t);
            }
            StreamEvent::Done(resp) => {
                println!("\nDone. Usage: {:?}", resp.usage);
                break;
            }
            _ => {}
        }
    }
    assert!(got_text);
}

#[tokio::test]
#[ignore] // Requires GOOGLE_API_KEY
async fn test_google_complete() {
    bit_ai::init();
    let model = bit_ai::model("google", "gemini-2.0-flash").unwrap();
    let req = ChatRequest::new(vec![Message::user("Say 'hello' and nothing else.")]);
    let resp = bit_ai::complete(&model, req).await.unwrap();

    assert!(!resp.message.content.is_empty());
    println!("Google response: {:?}", resp.message.content);
}

#[test]
fn test_init_registers_ollama() {
    bit_ai::init();
    // Ollama should always be registered (no API key needed)
    let result = bit_ai::model("ollama", "llama3.2");
    assert!(result.is_ok());
}

#[test]
fn test_init_without_env_vars() {
    // Should not panic even without any API keys
    bit_ai::init();
}
