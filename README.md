# aiform

Type-safe AI agents and tool calling for Rust.

Built on [`async-openai`](https://docs.rs/async-openai), providing type-safe tool definitions, agent execution loops, and multi-agent coordination.

## Why aiform?

- **Type-safe** - Tool schemas generated from your types at compile time
- **Simple** - Clean builder API, no complex abstractions
- **Fast** - Zero-cost abstractions, true async parallelism
- **Composable** - Agents are tools, tools are agents

## Install

```bash
cargo add aiform
```

## Quick Start

### Define Tools

```rust
use aiform::prelude::*;
use serde::Deserialize;

#[derive(ToolArg, Deserialize)]
struct WeatherArgs {
    location: String,
    unit: String,
}

#[tool("Get the current weather for a location")]
async fn get_weather(args: WeatherArgs) -> Result<String> {
    Ok(format!("Weather in {}: 22°{}", args.location, args.unit))
}
```

### Create an Agent

```rust
let agent = Agent::builder()
    .model("gpt-4")
    .system_prompt("You are a helpful weather assistant")
    .tools(tools![GetWeatherTool])
    .build()?;

let response = agent.run("What's the weather in Paris?").await?;
```

### Multi-turn Conversations

```rust
let mut conversation = Conversation::with_system("You are helpful");
conversation.add_user_message("Hello!");

let response = agent.run_conversation(&mut conversation).await?;
conversation.add_assistant_message(&response);

conversation.add_user_message("Tell me more");
let response = agent.run_conversation(&mut conversation).await?;
```

### Multi-Agent Patterns

```rust
// Specialized agents
let analyst = Agent::builder()
    .model("gpt-4")
    .system_prompt("You analyze data")
    .tools(tools![AnalyzeDataTool])
    .build()?;

let researcher = Agent::builder()
    .model("gpt-4")
    .system_prompt("You research topics")
    .tools(tools![SearchTool])
    .build()?;

// Researcher finds data, analyst analyzes it
let research = researcher.run("Find Rust adoption data").await?;
let analysis = analyst.call_as_tool(format!("Analyze: {}", research)).await?;
```

Agents can call other agents as tools, maintaining private contexts and only exposing final results.

## Features

- **Type-safe tool definitions** - `#[tool]` and `#[derive(ToolArg)]`
- **Agent execution loops** - Automatic tool calling and result handling
- **Multi-agent coordination** - Agents as tools, private conversations
- **Conversation management** - Track message history across turns
- **Error handling** - Comprehensive error types, no unwraps
- **Streaming support** - Coming soon

## Examples

See the [examples](examples/) directory:
- `simple_agent.rs` - Basic agent with tools
- `multi_agent.rs` - Multi-agent coordination patterns
- `openrouter_tools.rs` - Using with OpenRouter API

## Roadmap

- [ ] Streaming responses
- [ ] Agent teams and orchestration helpers
- [ ] Prompt templates
- [ ] Built-in retry logic
- [ ] Observability hooks

## License

MIT OR Apache-2.0
