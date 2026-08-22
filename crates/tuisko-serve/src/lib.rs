//! Exact OpenAI-compatible transport for the resident TuiskoLLM text server.

mod assistant;
mod request;

pub use assistant::{
    AssistantDelta, AssistantStreamFinish, AssistantStreamParser, ParsedAssistantOutput,
    ParsedToolCall, parse_assistant_output,
};
pub use request::{ChatCompletionRequest, ChatRequestError, PreparedChatRequest, SERVED_MODEL};
