use aiform::prelude::*;
use serde::Deserialize;

#[derive(ToolArg, Deserialize)]
struct CalculatorArgs {
    expression: String,
}

#[tool("Evaluate a mathematical expression")]
async fn calculate(args: CalculatorArgs) -> Result<String> {
    // Simple calculator - in production you'd use a proper parser
    let result = match args.expression.as_str() {
        expr if expr.contains('+') => {
            let parts: Vec<&str> = expr.split('+').collect();
            if parts.len() == 2 {
                let a: f64 = parts[0].trim().parse().unwrap_or(0.0);
                let b: f64 = parts[1].trim().parse().unwrap_or(0.0);
                a + b
            } else {
                0.0
            }
        }
        expr if expr.contains('*') => {
            let parts: Vec<&str> = expr.split('*').collect();
            if parts.len() == 2 {
                let a: f64 = parts[0].trim().parse().unwrap_or(0.0);
                let b: f64 = parts[1].trim().parse().unwrap_or(0.0);
                a * b
            } else {
                0.0
            }
        }
        _ => 0.0,
    };

    Ok(result.to_string())
}

#[tokio::main]
async fn main() -> Result<()> {
    // Create an agent with calculator tool
    let agent = Agent::builder()
        .model("gpt-4")
        .system_prompt(
            "You are a helpful math assistant. Use the calculator tool to solve problems.",
        )
        .tools(tools![CalculateTool])
        .build()?;

    // Single request
    println!("Single request example:");
    let response = agent.run("What is 15 + 27?").await?;
    println!("Agent: {}\n", response);

    // Multi-turn conversation
    println!("Multi-turn conversation example:");
    let mut conversation = Conversation::with_system(
        "You are a helpful math tutor. Guide students through problems step by step.",
    );

    conversation.add_user_message("I need to calculate 8 * 12");
    let response = agent.run_conversation(&mut conversation).await?;
    println!("Agent: {}", response);
    conversation.add_assistant_message(&response);

    conversation.add_user_message("Now divide that by 4");
    let response = agent.run_conversation(&mut conversation).await?;
    println!("Agent: {}", response);

    Ok(())
}
