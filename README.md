# aiform

Simple Rust macros for OpenAI-compatible tool calling and structured output.

Built on the [`async-openai`](https://docs.rs/async-openai) library.

## Install

```bash
cargo add aiform
```

## Example

```rust
use aiform::*;

#[derive(ToolArg, Deserialize)]
struct WeatherArgs {
    location: String,
    unit: String,
}

#[tool("Get the current weather for a location")]
async fn get_weather(args: WeatherArgs) -> Result<String, Box<dyn Error + Send + Sync>> {
    Ok(format!("Weather in {}: 22° {}", args.location, args.unit))
}

#[tokio::main]
async fn main() {
    let tools = tools![GetWeatherTool];
    
    let messages = vec![msg!(user "What's the weather in Paris?")];
    
    let request = CreateChatCompletionRequest {
        model: "gpt-4".to_string(),
        messages,
        tools: Some(tools.tools().to_vec()),
        ..Default::default()
    };
    
    let response = client.chat().create(request).await?;
    
    if let Some(tool_calls) = &response.choices[0].message.tool_calls {
        let results = dispatch_tool_calls(tool_calls, &tools).await?;
    }
}
```

## What it does

- `#[tool]` - turns async functions into OpenAI tools
- `#[derive(ToolArg)]` - generates JSON schemas for tool parameters
- `#[derive(StructuredOutput)]` - for structured output schemas
- `tools![]` - bundles tools with automatic dispatch
- `msg!()` - shorthand for chat messages

Works with OpenAI, OpenRouter, and any OpenAI-compatible API.
