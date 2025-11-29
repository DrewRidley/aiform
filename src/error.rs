//! Error types for the aiform library.

use std::fmt;

/// Result type alias using [`Error`].
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can occur when working with agents and tools.
#[derive(Debug)]
pub enum Error {
    /// An error occurred while calling the OpenAI API.
    OpenAI(async_openai::error::OpenAIError),

    /// An error occurred while serializing or deserializing JSON.
    Json(serde_json::Error),

    /// A tool with the given name was not found.
    ToolNotFound(String),

    /// An agent with the given name was not found.
    AgentNotFound(String),

    /// The agent execution loop exceeded the maximum number of iterations.
    MaxIterationsExceeded {
        /// The maximum number of iterations allowed.
        max: usize,
    },

    /// A tool execution failed.
    ToolExecution {
        /// The name of the tool that failed.
        tool_name: String,
        /// The underlying error message.
        message: String,
    },

    /// An invalid configuration was provided.
    InvalidConfiguration(String),

    /// A generic error occurred.
    Other(Box<dyn std::error::Error + Send + Sync>),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::OpenAI(e) => write!(f, "OpenAI API error: {}", e),
            Error::Json(e) => write!(f, "JSON error: {}", e),
            Error::ToolNotFound(name) => write!(f, "Tool not found: {}", name),
            Error::AgentNotFound(name) => write!(f, "Agent not found: {}", name),
            Error::MaxIterationsExceeded { max } => {
                write!(f, "Agent exceeded maximum iterations: {}", max)
            }
            Error::ToolExecution { tool_name, message } => {
                write!(f, "Tool '{}' failed: {}", tool_name, message)
            }
            Error::InvalidConfiguration(msg) => write!(f, "Invalid configuration: {}", msg),
            Error::Other(e) => write!(f, "{}", e),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::OpenAI(e) => Some(e),
            Error::Json(e) => Some(e),
            Error::Other(e) => Some(e.as_ref()),
            _ => None,
        }
    }
}

impl From<async_openai::error::OpenAIError> for Error {
    fn from(e: async_openai::error::OpenAIError) -> Self {
        Error::OpenAI(e)
    }
}

impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Error::Json(e)
    }
}

impl From<Box<dyn std::error::Error + Send + Sync>> for Error {
    fn from(e: Box<dyn std::error::Error + Send + Sync>) -> Self {
        Error::Other(e)
    }
}
