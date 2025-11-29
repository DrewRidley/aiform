//! Type-safe AI agents and tool calling for Rust.
//!
//! Built on top of the [`async_openai`](https://docs.rs/async-openai) library,
//! this crate provides:
//! - Type-safe tool definitions with automatic schema generation
//! - Agent execution loops with tool calling
//! - Multi-agent communication
//! - Conversation management
//!
//! Works with OpenAI, OpenRouter, and any OpenAI-compatible API.
//!
//! # Quick Start
//!
//! ## Defining Tools
//!
//! ```no_run
//! use aiform::prelude::*;
//! use serde::Deserialize;
//!
//! #[derive(ToolArg, Deserialize)]
//! struct WeatherArgs {
//!     location: String,
//!     unit: String,
//! }
//!
//! #[tool("Get the current weather for a location")]
//! async fn get_weather(args: WeatherArgs) -> Result<String> {
//!     Ok(format!("Weather in {}: 22°{}", args.location, args.unit))
//! }
//! ```
//!
//! ## Creating an Agent
//!
//! ```no_run
//! use aiform::prelude::*;
//! # use serde::Deserialize;
//! # #[derive(ToolArg, Deserialize)]
//! # struct WeatherArgs { location: String, unit: String }
//! # #[tool("Get weather")] async fn get_weather(args: WeatherArgs) -> Result<String> { Ok("".into()) }
//!
//! # async fn example() -> Result<()> {
//! let agent = Agent::builder()
//!     .model("gpt-4")
//!     .system_prompt("You are a helpful weather assistant")
//!     .tools(tools![GetWeatherTool])
//!     .build()?;
//!
//! let response = agent.run("What's the weather in Paris?").await?;
//! println!("{}", response);
//! # Ok(())
//! # }
//! ```
//!
//! ## Multi-turn Conversations
//!
//! ```no_run
//! use aiform::prelude::*;
//! # async fn example() -> Result<()> {
//! # let agent = Agent::builder().model("gpt-4").build()?;
//!
//! let mut conversation = Conversation::with_system("You are helpful");
//! conversation.add_user_message("Hello!");
//!
//! let response = agent.run_conversation(&mut conversation).await?;
//! conversation.add_assistant_message(&response);
//!
//! // Continue the conversation
//! conversation.add_user_message("Tell me more");
//! let response = agent.run_conversation(&mut conversation).await?;
//! # Ok(())
//! # }
//! ```

pub use aiform_macros::*;
pub use async_openai as openai;

pub mod agent;
pub mod agent_tool;
pub mod conversation;
pub mod error;

pub use agent::{Agent, AgentBuilder};
pub use agent_tool::AgentTool;
pub use conversation::Conversation;
pub use error::{Error, Result};

/// Convenience re-exports for common imports.
pub mod prelude {
    pub use crate::agent::{Agent, AgentBuilder};
    pub use crate::conversation::Conversation;
    pub use crate::error::{Error, Result};
    pub use crate::{msg, tool, tools, StructuredOutput, Tool, ToolArg, ToolSet};
}

/// Combines tool definitions with their dispatch logic.
///
/// Created using the `tools!` macro, this bundles OpenAI tool definitions
/// with a dispatcher that routes tool calls to their implementations.
pub struct ToolSet {
    /// The OpenAI tool definitions for API requests.
    pub tools: Vec<async_openai::types::ChatCompletionTool>,
    /// Dispatcher function that routes tool calls by name.
    pub dispatcher: Box<
        dyn Fn(
                String,
                serde_json::Value,
            ) -> std::pin::Pin<
                Box<
                    dyn std::future::Future<
                            Output = std::result::Result<
                                String,
                                Box<dyn std::error::Error + Send + Sync>,
                            >,
                        > + Send,
                >,
            > + Send
            + Sync,
    >,
}

impl ToolSet {
    /// Returns the tool definitions for use in API requests.
    pub fn tools(&self) -> &[async_openai::types::ChatCompletionTool] {
        &self.tools
    }

    /// Dispatches a tool call by name with the provided arguments.
    pub async fn dispatch(
        &self,
        name: String,
        args: serde_json::Value,
    ) -> std::result::Result<String, Box<dyn std::error::Error + Send + Sync>> {
        (self.dispatcher)(name, args).await
    }
}

impl Clone for ToolSet {
    fn clone(&self) -> Self {
        panic!("ToolSet cannot be cloned due to containing a closure. Use a reference instead.");
    }
}

/// Creates a `ToolSet` from tool structs.
///
/// # Example
///
/// ```ignore
/// let tools = tools![GetWeatherTool, CalculateTool];
/// ```
#[macro_export]
macro_rules! tools {
    ($($tool:ident),* $(,)?) => {{
        let tools_vec = vec![
            $(
                async_openai::types::ChatCompletionTool {
                    r#type: async_openai::types::ChatCompletionToolType::Function,
                    function: async_openai::types::FunctionObject {
                        name: $tool::NAME.to_string(),
                        description: Some($tool::DESCRIPTION.to_string()),
                        parameters: Some($tool::parameters()),
                    },
                },
            )*
        ];

        let dispatcher = Box::new(|name: String, args: serde_json::Value| {
            Box::pin(async move {
                match name.as_str() {
                    $(
                        $tool::NAME => {
                            $tool.call(args).await
                        }
                    )*
                    _ => Err("Unknown tool".into()),
                }
            }) as std::pin::Pin<Box<dyn std::future::Future<Output = std::result::Result<String, Box<dyn std::error::Error + Send + Sync>>> + Send>>
        });

        ToolSet {
            tools: tools_vec,
            dispatcher,
        }
    }};
}

/// Creates chat messages for OpenAI API requests.
///
/// # Examples
///
/// ```ignore
/// msg!(user "What's the weather?")
/// msg!(assistant "It's sunny")
/// msg!(assistant content, tool_calls)
/// msg!(tool tool_call_id, "result")
/// ```
#[macro_export]
macro_rules! msg {
    (user $content:expr) => {
        async_openai::types::ChatCompletionRequestMessage::User(
            async_openai::types::ChatCompletionRequestUserMessage {
                content: async_openai::types::ChatCompletionRequestUserMessageContent::Text(
                    $content.to_string(),
                ),
                role: async_openai::types::Role::User,
                name: None,
            },
        )
    };
    (assistant $content:expr) => {
        async_openai::types::ChatCompletionRequestMessage::Assistant(
            async_openai::types::ChatCompletionRequestAssistantMessage {
                content: Some($content.to_string()),
                tool_calls: None,
                ..Default::default()
            },
        )
    };
    (assistant $content:expr, $tool_calls:expr) => {
        async_openai::types::ChatCompletionRequestMessage::Assistant(
            async_openai::types::ChatCompletionRequestAssistantMessage {
                content: $content,
                tool_calls: $tool_calls,
                ..Default::default()
            },
        )
    };
    (tool $tool_call_id:expr, $content:expr) => {
        async_openai::types::ChatCompletionRequestMessage::Tool(
            async_openai::types::ChatCompletionRequestToolMessage {
                role: async_openai::types::Role::Tool,
                tool_call_id: $tool_call_id.to_string(),
                content: $content.to_string(),
            },
        )
    };
}

/// Dispatches multiple tool calls and returns their results.
///
/// Takes tool calls from an API response and executes them using the provided toolset.
pub async fn dispatch_tool_calls(
    tool_calls: &[async_openai::types::ChatCompletionMessageToolCall],
    toolset: &ToolSet,
) -> std::result::Result<Vec<String>, Box<dyn std::error::Error + Send + Sync>> {
    let mut results = vec![];
    for tool_call in tool_calls {
        let tool_name = tool_call.function.name.clone();
        let args: serde_json::Value = serde_json::from_str(&tool_call.function.arguments)?;
        let result = toolset.dispatch(tool_name, args).await?;
        results.push(result);
    }
    Ok(results)
}

/// Generates JSON schema for tool arguments.
///
/// Derive this on structs to use them as tool parameters.
pub trait ToolArg {
    /// Returns the JSON schema for this type.
    fn schema() -> serde_json::Value;
}

/// Trait implemented by the `#[tool]` macro.
///
/// Provides tool metadata and execution logic.
pub trait Tool {
    /// The tool's name.
    const NAME: &'static str;
    /// The tool's description.
    const DESCRIPTION: &'static str;
    /// Returns the JSON schema for the tool's parameters.
    fn parameters() -> serde_json::Value;
    /// Returns the tool's name.
    fn name() -> &'static str {
        Self::NAME
    }
    /// Returns the tool's description.
    fn description() -> &'static str {
        Self::DESCRIPTION
    }
    /// Executes the tool with the provided arguments.
    #[allow(async_fn_in_trait)]
    async fn call(
        &self,
        args: serde_json::Value,
    ) -> std::result::Result<String, Box<dyn std::error::Error + Send + Sync>>;
}

/// Generates JSON schema for structured output.
///
/// Derive this on structs to use them with OpenAI's structured output feature.
pub trait StructuredOutput {
    /// Returns the JSON schema for this type.
    fn schema() -> serde_json::Value;
}

/// Extension traits for the OpenAI client.
pub mod ext {
    use super::*;
    use async_openai::types::ChatCompletionRequestMessage;

    /// Extension methods for OpenAI client (not yet implemented).
    #[allow(async_fn_in_trait)]
    pub trait OpenAIClientExt {
        /// Makes a chat completion request with tools.
        async fn call_with_tools<T: Tool>(
            &self,
            messages: Vec<ChatCompletionRequestMessage>,
            tools: Vec<T>,
        ) -> std::result::Result<String, async_openai::error::OpenAIError>;

        /// Makes a chat completion request with structured output.
        async fn structured_output<S: StructuredOutput>(
            &self,
            messages: Vec<ChatCompletionRequestMessage>,
        ) -> std::result::Result<S, async_openai::error::OpenAIError>;
    }

    impl<C: async_openai::config::Config> OpenAIClientExt for async_openai::Client<C> {
        async fn call_with_tools<T: Tool>(
            &self,
            _messages: Vec<ChatCompletionRequestMessage>,
            _tools: Vec<T>,
        ) -> std::result::Result<String, async_openai::error::OpenAIError> {
            // Implementation would create the request with tools
            // For now, placeholder
            Err(async_openai::error::OpenAIError::InvalidArgument(
                "Not implemented".to_string(),
            ))
        }

        async fn structured_output<S: StructuredOutput>(
            &self,
            _messages: Vec<ChatCompletionRequestMessage>,
        ) -> std::result::Result<S, async_openai::error::OpenAIError> {
            // Implementation would use structured output
            // For now, placeholder
            Err(async_openai::error::OpenAIError::InvalidArgument(
                "Not implemented".to_string(),
            ))
        }
    }
}

#[cfg(test)]
#[allow(dead_code, unused_variables)]
mod tests {
    use super::*;
    use serde_json::json;

    #[derive(ToolArg, serde::Deserialize)]
    struct TestArgs {
        name: String,
        count: i32,
    }

    #[test]
    fn test_tool_arg_schema() {
        let schema = TestArgs::schema();
        assert_eq!(schema["type"], "object");

        assert_eq!(schema["properties"]["name"]["type"], "string");
        assert_eq!(schema["properties"]["count"]["type"], "integer");
        assert!(schema["required"]
            .as_array()
            .unwrap()
            .contains(&json!("name")));
        assert!(schema["required"]
            .as_array()
            .unwrap()
            .contains(&json!("count")));
    }

    #[tool("A test tool")]
    async fn test_tool(args: TestArgs) -> Result<String> {
        // Dummy implementation
        Ok(format!("Called with {} items", args.count))
    }

    #[test]
    fn test_tool_impl() {
        assert_eq!(TestToolTool::name(), "test_tool");
        assert_eq!(TestToolTool::description(), "A test tool");
        let params = TestToolTool::parameters();
        assert_eq!(params["type"], "object");
    }

    #[derive(StructuredOutput, ToolArg)]
    struct TestOutput {
        result: String,
    }

    #[test]
    fn test_structured_output_schema() {
        let schema = <TestOutput as ToolArg>::schema();
        assert_eq!(schema["type"], "object");
        assert_eq!(schema["properties"]["result"]["type"], "string");
    }

    #[derive(ToolArg)]
    struct Inner {
        value: i32,
    }

    #[derive(ToolArg)]
    struct Outer {
        inner: Inner,
        list: Vec<Inner>,
    }

    #[derive(ToolArg)]
    enum MyEnum {
        A,
        B,
        C,
    }

    #[derive(ToolArg)]
    enum WrappedEnum {
        Text(String),
        Number(i32),
    }

    #[derive(ToolArg)]
    enum ComplexEnum {
        Unit,
        Single(String),
        Multiple(String, i32),
        Named { text: String, num: i32 },
    }

    #[test]
    fn test_nested_schema() {
        let schema = Outer::schema();
        assert_eq!(schema["type"], "object");
        assert_eq!(schema["properties"]["inner"]["type"], "object");
        assert_eq!(
            schema["properties"]["inner"]["properties"]["value"]["type"],
            "integer"
        );
        assert_eq!(schema["properties"]["list"]["type"], "array");
        assert_eq!(schema["properties"]["list"]["items"]["type"], "object");
        assert_eq!(
            schema["properties"]["list"]["items"]["properties"]["value"]["type"],
            "integer"
        );
    }

    #[test]
    fn test_enum_schema() {
        let schema = MyEnum::schema();
        let one_of = schema["oneOf"].as_array().unwrap();
        assert_eq!(one_of.len(), 3);
        for item in one_of {
            assert_eq!(item["type"], "object");
            assert!(item["required"]
                .as_array()
                .unwrap()
                .contains(&json!("type")));
            let props = &item["properties"];
            assert!(props["type"]["const"].is_string());
        }
        // Check specific variants
        let types: std::collections::HashSet<_> = one_of
            .iter()
            .map(|item| item["properties"]["type"]["const"].as_str().unwrap())
            .collect();
        assert!(types.contains("A"));
        assert!(types.contains("B"));
        assert!(types.contains("C"));
    }

    #[test]
    fn test_wrapped_enum_schema() {
        let schema = WrappedEnum::schema();
        let one_of = schema["oneOf"].as_array().unwrap();
        assert_eq!(one_of.len(), 2);
        // Check Text variant
        let text_item = &one_of[0];
        assert_eq!(text_item["type"], "object");
        assert_eq!(text_item["properties"]["type"]["const"], "Text");
        assert_eq!(text_item["properties"]["value"]["type"], "string");
        assert!(text_item["required"]
            .as_array()
            .unwrap()
            .contains(&json!("type")));
        assert!(text_item["required"]
            .as_array()
            .unwrap()
            .contains(&json!("value")));
        // Check Number variant
        let number_item = &one_of[1];
        assert_eq!(number_item["type"], "object");
        assert_eq!(number_item["properties"]["type"]["const"], "Number");
        assert_eq!(number_item["properties"]["value"]["type"], "integer");
        assert!(number_item["required"]
            .as_array()
            .unwrap()
            .contains(&json!("type")));
        assert!(number_item["required"]
            .as_array()
            .unwrap()
            .contains(&json!("value")));
    }

    #[test]
    fn test_complex_enum_schema() {
        let schema = ComplexEnum::schema();
        let one_of = schema["oneOf"].as_array().unwrap();
        assert_eq!(one_of.len(), 4);

        // Unit variant
        let unit = &one_of[0];
        assert_eq!(unit["properties"]["type"]["const"], "Unit");
        assert_eq!(unit["required"].as_array().unwrap().len(), 1);

        // Single variant
        let single = &one_of[1];
        assert_eq!(single["properties"]["type"]["const"], "Single");
        assert_eq!(single["properties"]["value"]["type"], "string");
        assert!(single["required"]
            .as_array()
            .unwrap()
            .contains(&json!("value")));

        // Multiple variant
        let multiple = &one_of[2];
        assert_eq!(multiple["properties"]["type"]["const"], "Multiple");
        assert_eq!(multiple["properties"]["value"]["type"], "array");
        assert_eq!(
            multiple["properties"]["value"]["items"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
        assert!(multiple["required"]
            .as_array()
            .unwrap()
            .contains(&json!("value")));

        // Named variant
        let named = &one_of[3];
        assert_eq!(named["properties"]["type"]["const"], "Named");
        assert_eq!(named["properties"]["text"]["type"], "string");
        assert_eq!(named["properties"]["num"]["type"], "integer");
        let required = named["required"].as_array().unwrap();
        assert!(required.contains(&json!("text")));
        assert!(required.contains(&json!("num")));
    }
}
