//! Conversation and message management for agents.

use async_openai::types::{
    ChatCompletionRequestAssistantMessage, ChatCompletionRequestMessage,
    ChatCompletionRequestToolMessage, ChatCompletionRequestUserMessage,
    ChatCompletionRequestUserMessageContent, Role,
};

/// A conversation consisting of multiple messages.
///
/// Manages the message history for an agent, including user messages,
/// assistant responses, and tool call results.
#[derive(Debug, Clone, Default)]
pub struct Conversation {
    messages: Vec<ChatCompletionRequestMessage>,
}

impl Conversation {
    /// Creates a new empty conversation.
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
        }
    }

    /// Creates a conversation with a system message.
    pub fn with_system(system_prompt: impl Into<String>) -> Self {
        let mut conv = Self::new();
        conv.add_system_message(system_prompt);
        conv
    }

    /// Adds a system message to the conversation.
    pub fn add_system_message(&mut self, content: impl Into<String>) {
        self.messages.push(ChatCompletionRequestMessage::System(
            async_openai::types::ChatCompletionRequestSystemMessage {
                content: content.into(),
                role: Role::System,
                name: None,
            },
        ));
    }

    /// Adds a user message to the conversation.
    pub fn add_user_message(&mut self, content: impl Into<String>) {
        self.messages.push(ChatCompletionRequestMessage::User(
            ChatCompletionRequestUserMessage {
                content: ChatCompletionRequestUserMessageContent::Text(content.into()),
                role: Role::User,
                name: None,
            },
        ));
    }

    /// Adds an assistant message to the conversation.
    pub fn add_assistant_message(&mut self, content: impl Into<String>) {
        self.messages.push(ChatCompletionRequestMessage::Assistant(
            ChatCompletionRequestAssistantMessage {
                content: Some(content.into()),
                tool_calls: None,
                ..Default::default()
            },
        ));
    }

    /// Adds an assistant message with tool calls to the conversation.
    pub fn add_assistant_message_with_tools(
        &mut self,
        content: Option<String>,
        tool_calls: Vec<async_openai::types::ChatCompletionMessageToolCall>,
    ) {
        self.messages.push(ChatCompletionRequestMessage::Assistant(
            ChatCompletionRequestAssistantMessage {
                content,
                tool_calls: Some(tool_calls),
                ..Default::default()
            },
        ));
    }

    /// Adds a tool result message to the conversation.
    pub fn add_tool_message(
        &mut self,
        tool_call_id: impl Into<String>,
        content: impl Into<String>,
    ) {
        self.messages.push(ChatCompletionRequestMessage::Tool(
            ChatCompletionRequestToolMessage {
                role: Role::Tool,
                tool_call_id: tool_call_id.into(),
                content: content.into(),
            },
        ));
    }

    /// Returns a reference to all messages in the conversation.
    pub fn messages(&self) -> &[ChatCompletionRequestMessage] {
        &self.messages
    }

    /// Returns a mutable reference to all messages in the conversation.
    pub fn messages_mut(&mut self) -> &mut Vec<ChatCompletionRequestMessage> {
        &mut self.messages
    }

    /// Returns the number of messages in the conversation.
    pub fn len(&self) -> usize {
        self.messages.len()
    }

    /// Returns true if the conversation has no messages.
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    /// Clears all messages from the conversation.
    pub fn clear(&mut self) {
        self.messages.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conversation_creation() {
        let conv = Conversation::new();
        assert_eq!(conv.len(), 0);
        assert!(conv.is_empty());
    }

    #[test]
    fn test_with_system() {
        let conv = Conversation::with_system("You are helpful");
        assert_eq!(conv.len(), 1);
    }

    #[test]
    fn test_add_messages() {
        let mut conv = Conversation::new();
        conv.add_user_message("Hello");
        conv.add_assistant_message("Hi there!");

        assert_eq!(conv.len(), 2);
        assert!(!conv.is_empty());
    }

    #[test]
    fn test_clear() {
        let mut conv = Conversation::new();
        conv.add_user_message("Test");
        conv.clear();

        assert_eq!(conv.len(), 0);
        assert!(conv.is_empty());
    }
}
