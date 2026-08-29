//! Exact OpenAI-compatible transport for the resident TuiskoLLM text server.

mod assistant;
mod request;
mod request_log;
mod response;
mod server;
mod text_generator;

pub use assistant::{
    AssistantDelta, AssistantStreamFinish, AssistantStreamParser, ParsedAssistantOutput,
    ParsedToolCall, parse_assistant_output, parse_assistant_output_constrained,
};
pub use request::{ChatCompletionRequest, ChatRequestError, PreparedChatRequest, SERVED_MODEL};
pub use response::{GenerationReply, blocking_response, openai_error, streaming_response};
pub use server::{ServerConfig, ServerError, ServerModel, run};
