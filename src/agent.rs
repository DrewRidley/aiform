//! Agent implementation with tool execution and conversation management.

use crate::{
    conversation::Conversation,
    error::{Error, Result},
    ToolSet,
};
use async_openai::{types::CreateChatCompletionRequestArgs, Client};

/// Maximum number of agent loop iterations before stopping.
const DEFAULT_MAX_ITERATIONS: usize = 10;

/// An AI agent that can use tools and maintain conversations.
///
/// Agents execute a loop where they:
/// 1. Send messages to the LLM
/// 2. Receive a response (possibly with tool calls)
/// 3. Execute any requested tools
/// 4. Add tool results back to the conversation
/// 5. Repeat until the agent provides a final answer
///
/// # Example
///
/// ```no_run
/// use aiform::prelude::*;
/// # use async_openai::Client;
///
/// # async fn example() -> Result<()> {
/// let agent = Agent::builder()
///     .model("gpt-4")
///     .system_prompt("You are a helpful assistant")
///     .build()?;
///
/// let response = agent.run("Hello!").await?;
/// println!("{}", response);
/// # Ok(())
/// # }
/// ```
pub struct Agent {
    client: Client<async_openai::config::OpenAIConfig>,
    model: String,
    system_prompt: Option<String>,
    tools: Option<ToolSet>,
    max_iterations: usize,
}

impl Agent {
    /// Creates a new agent builder.
    pub fn builder() -> AgentBuilder {
        AgentBuilder::new()
    }

    /// Runs the agent with a single user message.
    ///
    /// This creates a new conversation with the given message and executes
    /// the agent loop until completion.
    ///
    /// # Errors
    ///
    /// Returns an error if the API call fails, tool execution fails, or
    /// the maximum number of iterations is exceeded.
    pub async fn run(&self, message: impl Into<String>) -> Result<String> {
        let mut conversation = if let Some(ref prompt) = self.system_prompt {
            Conversation::with_system(prompt.clone())
        } else {
            Conversation::new()
        };

        conversation.add_user_message(message);
        self.run_conversation(&mut conversation).await
    }

    /// Runs the agent with an existing conversation.
    ///
    /// This allows multi-turn conversations where the agent can reference
    /// previous messages.
    ///
    /// # Errors
    ///
    /// Returns an error if the API call fails, tool execution fails, or
    /// the maximum number of iterations is exceeded.
    pub async fn run_conversation(&self, conversation: &mut Conversation) -> Result<String> {
        self.execute_loop(conversation).await
    }

    /// Calls this agent as if it were a tool.
    ///
    /// This creates a fresh, isolated conversation for the request and returns
    /// only the final response. All intermediate tool calls and reasoning are
    /// hidden from the caller.
    ///
    /// This is used when one agent calls another agent.
    ///
    /// # Errors
    ///
    /// Returns an error if the agent execution fails.
    pub async fn call_as_tool(&self, message: impl Into<String>) -> Result<String> {
        let mut private_conversation = if let Some(ref prompt) = self.system_prompt {
            Conversation::with_system(prompt.clone())
        } else {
            Conversation::new()
        };

        private_conversation.add_user_message(message);
        self.execute_loop(&mut private_conversation).await
    }

    /// Executes the agent loop: LLM call -> tool execution -> repeat.
    async fn execute_loop(&self, conversation: &mut Conversation) -> Result<String> {
        for _iteration in 0..self.max_iterations {
            let mut request = CreateChatCompletionRequestArgs::default();
            request.model(&self.model);
            request.messages(conversation.messages().to_vec());

            if let Some(ref toolset) = self.tools {
                request.tools(toolset.tools().to_vec());
            }

            let request = request.build().map_err(|e| {
                Error::InvalidConfiguration(format!("Failed to build chat request: {}", e))
            })?;

            let response = self.client.chat().create(request).await?;

            let choice = response
                .choices
                .first()
                .ok_or_else(|| Error::Other("No response from API".into()))?;

            let message = &choice.message;

            // Check if there are tool calls
            if let Some(ref tool_calls) = message.tool_calls {
                // Add assistant message with tool calls
                conversation
                    .add_assistant_message_with_tools(message.content.clone(), tool_calls.clone());

                // Execute tools
                let toolset = self.tools.as_ref().ok_or_else(|| {
                    Error::InvalidConfiguration(
                        "Agent received tool calls but has no tools configured".to_string(),
                    )
                })?;

                for tool_call in tool_calls {
                    let tool_name = &tool_call.function.name;
                    let args: serde_json::Value =
                        serde_json::from_str(&tool_call.function.arguments)?;

                    let result = toolset
                        .dispatch(tool_name.clone(), args)
                        .await
                        .map_err(|e| Error::ToolExecution {
                            tool_name: tool_name.clone(),
                            message: e.to_string(),
                        })?;

                    conversation.add_tool_message(&tool_call.id, result);
                }

                // Continue the loop to get the next response
                continue;
            }

            // No tool calls, this is the final response
            if let Some(content) = &message.content {
                return Ok(content.clone());
            }

            return Err(Error::Other(
                "Agent returned no content or tool calls".into(),
            ));
        }

        Err(Error::MaxIterationsExceeded {
            max: self.max_iterations,
        })
    }
}

/// Builder for creating agents.
///
/// # Example
///
/// ```no_run
/// use aiform::prelude::*;
/// # use async_openai::Client;
///
/// # async fn example() -> Result<()> {
/// let agent = Agent::builder()
///     .model("gpt-4")
///     .system_prompt("You are a helpful coding assistant")
///     .max_iterations(15)
///     .build()?;
/// # Ok(())
/// # }
/// ```
pub struct AgentBuilder {
    client: Option<Client<async_openai::config::OpenAIConfig>>,
    model: Option<String>,
    system_prompt: Option<String>,
    tools: Option<ToolSet>,
    max_iterations: Option<usize>,
}

impl AgentBuilder {
    /// Creates a new agent builder with default settings.
    pub fn new() -> Self {
        Self {
            client: None,
            model: None,
            system_prompt: None,
            tools: None,
            max_iterations: None,
        }
    }

    /// Sets the OpenAI client to use.
    ///
    /// If not set, a default client will be created.
    pub fn client(mut self, client: Client<async_openai::config::OpenAIConfig>) -> Self {
        self.client = Some(client);
        self
    }

    /// Sets the model to use (e.g., "gpt-4", "gpt-3.5-turbo").
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Sets the system prompt for the agent.
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Sets the tools available to the agent.
    pub fn tools(mut self, tools: ToolSet) -> Self {
        self.tools = Some(tools);
        self
    }

    /// Sets the maximum number of iterations for the agent loop.
    ///
    /// Default is 10 iterations.
    pub fn max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = Some(max);
        self
    }

    /// Builds the agent.
    ///
    /// # Errors
    ///
    /// Returns an error if required fields (model) are not set.
    pub fn build(self) -> Result<Agent> {
        let model = self
            .model
            .ok_or_else(|| Error::InvalidConfiguration("Model must be specified".to_string()))?;

        let client = self.client.unwrap_or_else(Client::new);

        Ok(Agent {
            client,
            model,
            system_prompt: self.system_prompt,
            tools: self.tools,
            max_iterations: self.max_iterations.unwrap_or(DEFAULT_MAX_ITERATIONS),
        })
    }
}

impl Default for AgentBuilder {
    fn default() -> Self {
        Self::new()
    }
}
