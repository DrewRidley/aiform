use aiform::*;
use async_openai::{config::OpenAIConfig, types::CreateChatCompletionRequest, Client};
use serde::Deserialize;
use std::env;

#[derive(ToolArg, Deserialize)]
#[allow(dead_code)]
struct WeatherArgs {
    location: String,
    unit: String,
}

#[tool("Get the current weather for a location")]
async fn get_weather(
    args: WeatherArgs,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    // Simulate weather API call
    let unit = match args.unit.as_str() {
        "celsius" => "C",
        "fahrenheit" => "F",
        _ => "C",
    };
    Ok(format!("Weather in {}: 22° {}", args.location, unit))
}

#[derive(ToolArg, Deserialize)]
#[allow(dead_code)]
struct CalculatorArgs {
    expression: String,
}

#[tool("Evaluate a mathematical expression")]
async fn calculate(
    args: CalculatorArgs,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    // Simple calculator simulation
    match args.expression.as_str() {
        "2 + 2" => Ok("4".to_string()),
        "10 * 5" => Ok("50".to_string()),
        _ => Ok(format!("Calculated: {}", args.expression)),
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Set up OpenRouter client
    let api_key = env::var("OPENROUTER_API_KEY")
        .expect("OPENROUTER_API_KEY environment variable must be set");

    let config = OpenAIConfig::new()
        .with_api_key(api_key)
        .with_api_base("https://openrouter.ai/api/v1");

    let client = Client::with_config(config);

    // Define available tools
    let tools = tools![GetWeatherTool, CalculateTool];

    // Initial conversation
    let mut messages =
        vec![msg!(user "What's the weather in Paris in Celsius, and what's 15 * 7?")];

    // Tool call loop
    loop {
        let request = CreateChatCompletionRequest {
            model: "qwen/qwen3-32b:nitro".to_string(), // OpenRouter model
            messages: messages.clone(),
            tools: Some(tools.tools().to_vec()),
            ..Default::default()
        };

        let response = client.chat().create(request).await?;
        let choice = response.choices.first().unwrap();
        let assistant_message = &choice.message;

        // Add assistant message to conversation
        messages.push(
            msg!(assistant assistant_message.content.clone(), assistant_message.tool_calls.clone()),
        );

        // Check if there are tool calls
        if let Some(tool_calls) = &assistant_message.tool_calls {
            let results = dispatch_tool_calls(tool_calls, &tools).await?;

            for (tool_call, result) in tool_calls.iter().zip(results) {
                messages.push(msg!(tool tool_call.id, result));
            }
        } else {
            // No more tool calls, print final response
            if let Some(content) = &assistant_message.content {
                println!("Final response: {}", content);
            }
            break;
        }
    }

    Ok(())
}
