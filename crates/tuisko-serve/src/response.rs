//! Blocking and server-sent-event OpenAI response bodies.

use crate::{AssistantDelta, AssistantStreamParser, ParsedToolCall, SERVED_MODEL};
use axum::Json;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use serde::Serialize;
use serde_json::{Value, json};
use std::convert::Infallible;
use tokio::sync::mpsc::{UnboundedReceiver, unbounded_channel};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tuisko_engine::GeneratedText;

/// One exact scheduler-to-HTTP reply.
pub enum GenerationReply {
    /// Newly decoded text.
    Delta(String),
    /// Terminal generated output and accounting.
    Done(GeneratedText),
    /// Request admission failed before any device work was scheduled.
    Rejected(String),
    /// Resident execution failed after admission.
    Failed(String),
}

#[derive(Serialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

/// Collects one generation channel into an OpenAI blocking response.
pub async fn blocking_response(
    mut replies: UnboundedReceiver<GenerationReply>,
    id: String,
    created: u64,
    split_reasoning: bool,
    parse_tools: bool,
) -> Response {
    let mut text = String::new();
    while let Some(reply) = replies.recv().await {
        match reply {
            GenerationReply::Delta(delta) => text.push_str(&delta),
            GenerationReply::Done(output) => {
                debug_assert_eq!(text, output.text, "streamed and terminal text diverged");
                let parsed = crate::parse_assistant_output(&text, split_reasoning, parse_tools);
                let mut message = json!({
                    "role": "assistant",
                    "content": if parsed.tool_calls.is_empty() || !parsed.content.is_empty() {
                        Value::String(parsed.content)
                    } else {
                        Value::Null
                    }
                });
                if split_reasoning {
                    message["reasoning_content"] = Value::String(parsed.reasoning);
                }
                if !parsed.tool_calls.is_empty() {
                    message["tool_calls"] = blocking_tool_calls(&id, &parsed.tool_calls);
                }
                let finish_reason = if parsed.tool_calls.is_empty() {
                    output.finish_reason.as_str()
                } else {
                    "tool_calls"
                };
                let usage = usage(&output);
                return Json(json!({
                    "id": id,
                    "object": "chat.completion",
                    "created": created,
                    "model": SERVED_MODEL,
                    "choices": [{
                        "index": 0,
                        "message": message,
                        "finish_reason": finish_reason
                    }],
                    "usage": usage
                }))
                .into_response();
            }
            GenerationReply::Rejected(message) => {
                return openai_error(StatusCode::BAD_REQUEST, message, "invalid_request_error");
            }
            GenerationReply::Failed(message) => {
                return openai_error(StatusCode::INTERNAL_SERVER_ERROR, message, "server_error");
            }
        }
    }
    openai_error(
        StatusCode::SERVICE_UNAVAILABLE,
        "resident engine worker disconnected".into(),
        "server_error",
    )
}

/// Streams one generation channel as OpenAI server-sent events followed by `[DONE]`.
pub fn streaming_response(
    mut replies: UnboundedReceiver<GenerationReply>,
    id: String,
    created: u64,
    split_reasoning: bool,
    parse_tools: bool,
) -> Response {
    let (events_tx, events_rx) = unbounded_channel::<Result<Event, Infallible>>();
    tokio::spawn(async move {
        let first = json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": SERVED_MODEL,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]
        });
        if events_tx
            .send(Ok(Event::default().data(first.to_string())))
            .is_err()
        {
            return;
        }

        let mut parser = AssistantStreamParser::new(split_reasoning, parse_tools);
        let mut terminal = false;
        while let Some(reply) = replies.recv().await {
            match reply {
                GenerationReply::Delta(delta) => {
                    let parsed = parser.push(&delta);
                    if let Some(event) = assistant_delta_event(&id, created, parsed)
                        && events_tx.send(Ok(event)).is_err()
                    {
                        return;
                    }
                }
                GenerationReply::Done(output) => {
                    terminal = true;
                    let parsed = parser.finish();
                    if let Some(event) = assistant_delta_event(&id, created, parsed.delta)
                        && events_tx.send(Ok(event)).is_err()
                    {
                        return;
                    }
                    if let Some(event) = tool_calls_event(&id, created, &parsed.tool_calls)
                        && events_tx.send(Ok(event)).is_err()
                    {
                        return;
                    }
                    let finish_reason = if parsed.tool_calls.is_empty() {
                        output.finish_reason.as_str()
                    } else {
                        "tool_calls"
                    };
                    let event = json!({
                        "id": id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": SERVED_MODEL,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                        "usage": usage(&output)
                    });
                    if events_tx
                        .send(Ok(Event::default().data(event.to_string())))
                        .is_err()
                    {
                        return;
                    }
                    break;
                }
                error => {
                    terminal = true;
                    let (message, error_type) = match error {
                        GenerationReply::Rejected(message) => (message, "invalid_request_error"),
                        GenerationReply::Failed(message) => (message, "server_error"),
                        _ => unreachable!("delta and terminal replies were handled above"),
                    };
                    let event = json!({
                        "error": {"message": message, "type": error_type}
                    });
                    if events_tx
                        .send(Ok(Event::default().data(event.to_string())))
                        .is_err()
                    {
                        return;
                    }
                    break;
                }
            }
        }
        if !terminal {
            let event = json!({
                "error": {
                    "message": "resident engine worker disconnected",
                    "type": "server_error"
                }
            });
            if events_tx
                .send(Ok(Event::default().data(event.to_string())))
                .is_err()
            {
                return;
            }
        }
        let _ = events_tx.send(Ok(Event::default().data("[DONE]")));
    });
    let stream = UnboundedReceiverStream::new(events_rx);
    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

/// Builds one OpenAI error envelope with the supplied status and type.
pub fn openai_error(status: StatusCode, message: String, error_type: &'static str) -> Response {
    (
        status,
        Json(json!({
            "error": {"message": message, "type": error_type, "param": null, "code": null}
        })),
    )
        .into_response()
}

fn usage(output: &GeneratedText) -> Usage {
    Usage {
        prompt_tokens: output.prompt.token_ids.len(),
        completion_tokens: output.token_ids.len(),
        total_tokens: output.prompt.token_ids.len() + output.token_ids.len(),
    }
}

fn tool_call_id(completion_id: &str, index: usize) -> String {
    let suffix = completion_id
        .strip_prefix("chatcmpl-tuisko-")
        .unwrap_or(completion_id);
    format!("call_tuisko_{suffix}_{index}")
}

fn blocking_tool_calls(completion_id: &str, calls: &[ParsedToolCall]) -> Value {
    Value::Array(
        calls
            .iter()
            .enumerate()
            .map(|(index, call)| {
                json!({
                    "id": tool_call_id(completion_id, index),
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments}
                })
            })
            .collect(),
    )
}

fn tool_calls_event(id: &str, created: u64, calls: &[ParsedToolCall]) -> Option<Event> {
    if calls.is_empty() {
        return None;
    }
    let event = json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": SERVED_MODEL,
        "choices": [{
            "index": 0,
            "delta": {
                "tool_calls": calls.iter().enumerate().map(|(index, call)| json!({
                    "index": index,
                    "id": tool_call_id(id, index),
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments}
                })).collect::<Vec<_>>()
            },
            "finish_reason": null
        }]
    });
    Some(Event::default().data(event.to_string()))
}

fn assistant_delta_event(id: &str, created: u64, parsed: AssistantDelta) -> Option<Event> {
    let delta = match (parsed.reasoning.is_empty(), parsed.content.is_empty()) {
        (true, true) => return None,
        (false, true) => json!({"reasoning_content": parsed.reasoning}),
        (true, false) => json!({"content": parsed.content}),
        (false, false) => {
            json!({"reasoning_content": parsed.reasoning, "content": parsed.content})
        }
    };
    let event = json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": SERVED_MODEL,
        "choices": [{"index": 0, "delta": delta, "finish_reason": null}]
    });
    Some(Event::default().data(event.to_string()))
}

#[cfg(test)]
mod tests {
    use super::{GenerationReply, blocking_response, streaming_response};
    use axum::body::to_bytes;
    use axum::http::{StatusCode, header};
    use serde_json::{Value, json};
    use tokio::runtime::Builder;
    use tokio::sync::mpsc::unbounded_channel;
    use tuisko_engine::{FinishReason, GeneratedText};
    use tuisko_frontend::PromptEncoding;

    fn output(text: &str, reason: FinishReason) -> GeneratedText {
        GeneratedText {
            prompt: PromptEncoding {
                token_ids: vec![10, 11, 12],
                reused_tokens: 2,
                rendered_bytes: 9,
                fresh_bytes: 3,
            },
            token_ids: vec![20, 21],
            text: text.into(),
            finish_reason: reason,
        }
    }

    async fn body(response: axum::response::Response) -> String {
        let bytes = to_bytes(response.into_body(), 1 << 20).await.unwrap();
        String::from_utf8(bytes.to_vec()).unwrap()
    }

    fn runtime() -> tokio::runtime::Runtime {
        Builder::new_current_thread().enable_all().build().unwrap()
    }

    #[test]
    fn blocking_response_carries_reasoning_content_and_exact_usage() {
        runtime().block_on(async {
            let (sender, receiver) = unbounded_channel();
            sender
                .send(GenerationReply::Delta("think</think>\n\nanswer".into()))
                .unwrap();
            sender
                .send(GenerationReply::Done(output(
                    "think</think>\n\nanswer",
                    FinishReason::Length,
                )))
                .unwrap();
            let response =
                blocking_response(receiver, "chatcmpl-tuisko-0001".into(), 17, true, false).await;

            assert_eq!(response.status(), StatusCode::OK);
            let value: Value = serde_json::from_str(&body(response).await).unwrap();
            assert_eq!(value["choices"][0]["message"]["content"], "answer");
            assert_eq!(value["choices"][0]["message"]["reasoning_content"], "think");
            assert_eq!(value["choices"][0]["finish_reason"], "length");
            assert_eq!(
                value["usage"],
                json!({"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5})
            );
        });
    }

    #[test]
    fn streaming_response_emits_role_tool_usage_and_done_events() {
        runtime().block_on(async {
            let (sender, receiver) = unbounded_channel();
            sender
                .send(GenerationReply::Delta(
                    "inspect</think>\n\n<tool_".into(),
                ))
                .unwrap();
            sender
                .send(GenerationReply::Delta(
                    "call><function=bash><parameter=command>ls</parameter></function></tool_call>"
                        .into(),
                ))
                .unwrap();
            sender
                .send(GenerationReply::Done(output(
                    "inspect</think>\n\n<tool_call><function=bash><parameter=command>ls</parameter></function></tool_call>",
                    FinishReason::Stop,
                )))
                .unwrap();
            let response = streaming_response(
                receiver,
                "chatcmpl-tuisko-0002".into(),
                19,
                true,
                true,
            );

            assert_eq!(
                response.headers()[header::CONTENT_TYPE],
                "text/event-stream"
            );
            let body = body(response).await;
            let data = body
                .lines()
                .filter_map(|line| line.strip_prefix("data: "))
                .collect::<Vec<_>>();
            assert_eq!(data.last(), Some(&"[DONE]"));
            let events = data[..data.len() - 1]
                .iter()
                .map(|event| serde_json::from_str::<Value>(event).unwrap())
                .collect::<Vec<_>>();

            assert_eq!(events[0]["choices"][0]["delta"]["role"], "assistant");
            assert_eq!(
                events[1]["choices"][0]["delta"]["reasoning_content"],
                "inspect"
            );
            assert_eq!(
                events[2]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"],
                r#"{"command":"ls"}"#
            );
            assert_eq!(events[3]["choices"][0]["finish_reason"], "tool_calls");
            assert_eq!(events[3]["usage"]["total_tokens"], 5);
        });
    }

    #[test]
    fn blocking_disconnect_and_engine_error_are_distinct() {
        runtime().block_on(async {
            let (sender, receiver) = unbounded_channel();
            sender
                .send(GenerationReply::Rejected("bad sampling".into()))
                .unwrap();
            let response = blocking_response(receiver, "id".into(), 1, false, false).await;
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            let error: Value = serde_json::from_str(&body(response).await).unwrap();
            assert_eq!(error["error"]["message"], "bad sampling");

            let (sender, receiver) = unbounded_channel();
            sender
                .send(GenerationReply::Failed("device launch failed".into()))
                .unwrap();
            let response = blocking_response(receiver, "id".into(), 1, false, false).await;
            assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
            let error: Value = serde_json::from_str(&body(response).await).unwrap();
            assert_eq!(error["error"]["type"], "server_error");

            let (sender, receiver) = unbounded_channel::<GenerationReply>();
            drop(sender);
            let response = blocking_response(receiver, "id".into(), 1, false, false).await;
            assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        });
    }

    #[test]
    fn streaming_rejection_and_disconnect_are_distinct() {
        runtime().block_on(async {
            let (sender, receiver) = unbounded_channel();
            sender
                .send(GenerationReply::Rejected("context is full".into()))
                .unwrap();
            let response = streaming_response(receiver, "id".into(), 1, false, false);
            let rejection_body = body(response).await;
            let data = rejection_body
                .lines()
                .filter_map(|line| line.strip_prefix("data: "))
                .collect::<Vec<_>>();
            let error: Value = serde_json::from_str(data[data.len() - 2]).unwrap();
            assert_eq!(error["error"]["type"], "invalid_request_error");

            let (sender, receiver) = unbounded_channel::<GenerationReply>();
            drop(sender);
            let response = streaming_response(receiver, "id".into(), 1, false, false);
            let body = body(response).await;
            let data = body
                .lines()
                .filter_map(|line| line.strip_prefix("data: "))
                .collect::<Vec<_>>();

            assert_eq!(data.last(), Some(&"[DONE]"));
            let error: Value = serde_json::from_str(data[data.len() - 2]).unwrap();
            assert_eq!(error["error"]["type"], "server_error");
        });
    }
}
