//! Utilities for using agents as tools.

use crate::{Agent, Tool, ToolArg};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;

/// Arguments for calling an agent as a tool.
#[derive(Debug, Clone, Serialize, Deserialize, ToolArg)]
pub struct AgentCallArgs {
    /// The message to send to the agent.
    pub message: String,
}

/// Wrapper that allows an agent to be used as a tool.
///
/// This enables multi-agent patterns where one agent can delegate work
/// to another specialized agent. The calling agent only sees the final
/// response, not the intermediate tool calls or reasoning.
///
/// # Example
///
/// ```ignore
/// use aiform::prelude::*;
/// use aiform::agent_tool::AgentTool;
/// use std::sync::Arc;
/// use tokio::sync::Mutex;
///
/// # async fn example() -> Result<()> {
/// // Create a specialized analyst agent
/// let analyst = Agent::builder()
///     .model("gpt-4")
///     .system_prompt("You analyze data and provide insights")
///     .build()?;
///
/// let analyst_tool = AgentTool::new(
///     "analyst",
///     "Ask the analyst agent to analyze data",
///     Arc::new(Mutex::new(analyst)),
/// );
///
/// // In the future, you'll be able to use agents as tools directly
/// // For now, you can call the analyst manually
/// let response = analyst_tool.call_agent("Analyze this data").await?;
/// # Ok(())
/// # }
/// ```
#[allow(dead_code)]
pub struct AgentTool {
    name: String,
    description: String,
    agent: Arc<Mutex<Agent>>,
}

impl AgentTool {
    /// Creates a new agent tool.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the tool (how it appears to the calling agent)
    /// * `description` - Description of what the agent does
    /// * `agent` - The agent to wrap
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        agent: Arc<Mutex<Agent>>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            agent,
        }
    }
}

impl Tool for AgentTool {
    const NAME: &'static str = "agent_call";
    const DESCRIPTION: &'static str = "Call another agent";

    fn name() -> &'static str {
        Self::NAME
    }

    fn description() -> &'static str {
        Self::DESCRIPTION
    }

    fn parameters() -> serde_json::Value {
        AgentCallArgs::schema()
    }

    async fn call(
        &self,
        args: serde_json::Value,
    ) -> std::result::Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let args: AgentCallArgs = serde_json::from_value(args)?;
        let agent = self.agent.lock().await;
        let response = agent.call_as_tool(args.message).await?;
        Ok(response)
    }
}
