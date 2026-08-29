//! Incremental Qwen reasoning and tool-call parsing.

use serde_json::{Map, Value};
use std::sync::Arc;
use tuisko_frontend::ToolCallConstraintSpec;

const THINK_END: &str = "</think>";
const CALL_OPEN: &str = "<tool_call>";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AssistantPhase {
    Reasoning,
    Content,
}

/// Reasoning and user-visible text newly completed by one parser push.
#[derive(Debug, Default, Eq, PartialEq)]
pub struct AssistantDelta {
    /// Newly complete reasoning text.
    pub reasoning: String,
    /// Newly complete assistant content.
    pub content: String,
}

/// One parsed OpenAI-compatible function call.
#[derive(Debug, Eq, PartialEq)]
pub struct ParsedToolCall {
    /// Function name emitted by the checkpoint.
    pub name: String,
    /// Function arguments encoded as one JSON object string.
    pub arguments: String,
}

/// Complete parsed assistant response.
#[derive(Debug, Default, Eq, PartialEq)]
pub struct ParsedAssistantOutput {
    /// Complete reasoning text.
    pub reasoning: String,
    /// Complete user-visible content.
    pub content: String,
    /// Structured tool calls, when a complete Qwen tool span was emitted.
    pub tool_calls: Vec<ParsedToolCall>,
    /// Whether a constrained tool span violated the generation invariant.
    pub malformed_tool_call: bool,
    /// Whether the hard token limit ended an incomplete buffered tool span.
    pub truncated_tool_call: bool,
}

/// Final delayed text and parsed calls produced when a stream ends.
#[derive(Debug, Default, Eq, PartialEq)]
pub struct AssistantStreamFinish {
    /// Text withheld while a partial marker was unresolved.
    pub delta: AssistantDelta,
    /// Structured calls retained until their closing tags were available.
    pub tool_calls: Vec<ParsedToolCall>,
    /// Whether a constrained tool span violated the generation invariant.
    pub malformed_tool_call: bool,
    /// Whether the hard token limit ended an incomplete buffered tool span.
    pub truncated_tool_call: bool,
}

struct ReasoningStreamParser {
    phase: AssistantPhase,
    pending: String,
    strip_content_newlines: bool,
}

impl ReasoningStreamParser {
    fn new(split_reasoning: bool) -> Self {
        Self {
            phase: if split_reasoning {
                AssistantPhase::Reasoning
            } else {
                AssistantPhase::Content
            },
            pending: String::new(),
            strip_content_newlines: false,
        }
    }

    fn push(&mut self, text: &str) -> AssistantDelta {
        if self.phase == AssistantPhase::Content {
            return AssistantDelta {
                content: self.content_delta(text),
                ..AssistantDelta::default()
            };
        }

        self.pending.push_str(text);
        if let Some(boundary) = self.pending.find(THINK_END) {
            let reasoning = self.pending[..boundary].to_owned();
            let content_begin = boundary + THINK_END.len();
            let content = self.pending[content_begin..].to_owned();
            self.pending.clear();
            self.phase = AssistantPhase::Content;
            self.strip_content_newlines = true;
            return AssistantDelta {
                reasoning,
                content: self.content_delta(&content),
            };
        }

        let retained = retained_marker_prefix(&self.pending, THINK_END);
        let retained_text = self.pending.split_off(self.pending.len() - retained);
        let reasoning = std::mem::replace(&mut self.pending, retained_text);
        AssistantDelta {
            reasoning,
            ..AssistantDelta::default()
        }
    }

    fn finish(&mut self) -> AssistantDelta {
        if self.phase == AssistantPhase::Reasoning {
            AssistantDelta {
                reasoning: std::mem::take(&mut self.pending),
                ..AssistantDelta::default()
            }
        } else {
            AssistantDelta::default()
        }
    }

    fn content_delta(&mut self, text: &str) -> String {
        if !self.strip_content_newlines {
            return text.to_owned();
        }
        let text = text.trim_start_matches(['\r', '\n']);
        if !text.is_empty() {
            self.strip_content_newlines = false;
        }
        text.to_owned()
    }
}

/// Incrementally separates reasoning and content while retaining Qwen tool XML until complete.
pub struct AssistantStreamParser {
    reasoning: ReasoningStreamParser,
    parse_tools: bool,
    pending_marker: String,
    tool_text: Option<String>,
    tool_constraint: Option<Arc<ToolCallConstraintSpec>>,
}

impl AssistantStreamParser {
    /// Creates a parser with the response modes selected during request admission.
    pub fn new(split_reasoning: bool, parse_tools: bool) -> Self {
        Self {
            reasoning: ReasoningStreamParser::new(split_reasoning),
            parse_tools,
            pending_marker: String::new(),
            tool_text: None,
            tool_constraint: None,
        }
    }

    /// Creates a parser that validates generated calls against the request contract.
    pub fn with_constraint(
        split_reasoning: bool,
        tool_constraint: Arc<ToolCallConstraintSpec>,
    ) -> Self {
        Self {
            reasoning: ReasoningStreamParser::new(split_reasoning),
            parse_tools: true,
            pending_marker: String::new(),
            tool_text: None,
            tool_constraint: Some(tool_constraint),
        }
    }

    /// Consumes one decoded generation delta.
    pub fn push(&mut self, text: &str) -> AssistantDelta {
        if !self.parse_tools {
            return self.reasoning.push(text);
        }
        if let Some(tool_text) = &mut self.tool_text {
            tool_text.push_str(text);
            return AssistantDelta::default();
        }

        self.pending_marker.push_str(text);
        if let Some(marker) = self.pending_marker.find(CALL_OPEN) {
            let tool_text = self.pending_marker[marker..].to_owned();
            let prefix = self.pending_marker[..marker].to_owned();
            self.pending_marker.clear();
            self.tool_text = Some(tool_text);
            return self.reasoning.push(&prefix);
        }

        let retained = retained_marker_prefix(&self.pending_marker, CALL_OPEN);
        let retained_text = self
            .pending_marker
            .split_off(self.pending_marker.len() - retained);
        let publish = std::mem::replace(&mut self.pending_marker, retained_text);
        self.reasoning.push(&publish)
    }

    /// Flushes partial markers and returns any complete structured calls.
    pub fn finish(&mut self) -> AssistantStreamFinish {
        self.finish_with_truncation(false)
    }

    /// Flushes the stream, discarding an incomplete tool span at the hard token limit.
    pub fn finish_with_truncation(&mut self, truncated: bool) -> AssistantStreamFinish {
        if let Some(tool_text) = self.tool_text.take() {
            if let Some((prefix, tool_calls)) =
                parse_qwen_tool_calls(&tool_text, self.tool_constraint.as_deref())
            {
                let mut delta = self.reasoning.push(&prefix);
                append_delta(&mut delta, self.reasoning.finish());
                return AssistantStreamFinish {
                    delta,
                    tool_calls,
                    malformed_tool_call: false,
                    truncated_tool_call: false,
                };
            }
            let structurally_complete = parse_qwen_tool_calls(&tool_text, None).is_some();
            if truncated && !structurally_complete {
                let delta = self.reasoning.finish();
                return AssistantStreamFinish {
                    delta,
                    tool_calls: Vec::new(),
                    malformed_tool_call: false,
                    truncated_tool_call: true,
                };
            }
            if self.tool_constraint.is_some() {
                let delta = self.reasoning.finish();
                return AssistantStreamFinish {
                    delta,
                    tool_calls: Vec::new(),
                    malformed_tool_call: true,
                    truncated_tool_call: false,
                };
            }
            let mut delta = self.reasoning.push(&tool_text);
            append_delta(&mut delta, self.reasoning.finish());
            return AssistantStreamFinish {
                delta,
                tool_calls: Vec::new(),
                malformed_tool_call: false,
                truncated_tool_call: false,
            };
        }

        let pending = std::mem::take(&mut self.pending_marker);
        let mut delta = self.reasoning.push(&pending);
        append_delta(&mut delta, self.reasoning.finish());
        AssistantStreamFinish {
            delta,
            tool_calls: Vec::new(),
            malformed_tool_call: false,
            truncated_tool_call: false,
        }
    }
}

/// Parses one complete decoded generation using the same incremental path used by SSE.
pub fn parse_assistant_output(
    text: &str,
    split_reasoning: bool,
    parse_tools: bool,
) -> ParsedAssistantOutput {
    let mut parser = AssistantStreamParser::new(split_reasoning, parse_tools);
    let first = parser.push(text);
    let tail = parser.finish();
    ParsedAssistantOutput {
        reasoning: first.reasoning + &tail.delta.reasoning,
        content: first.content + &tail.delta.content,
        tool_calls: tail.tool_calls,
        malformed_tool_call: tail.malformed_tool_call,
        truncated_tool_call: tail.truncated_tool_call,
    }
}

/// Parses one complete constrained generation, retaining hard-limit truncation separately.
pub fn parse_assistant_output_constrained(
    text: &str,
    split_reasoning: bool,
    tool_constraint: Arc<ToolCallConstraintSpec>,
    truncated: bool,
) -> ParsedAssistantOutput {
    let mut parser = AssistantStreamParser::with_constraint(split_reasoning, tool_constraint);
    let first = parser.push(text);
    let tail = parser.finish_with_truncation(truncated);
    ParsedAssistantOutput {
        reasoning: first.reasoning + &tail.delta.reasoning,
        content: first.content + &tail.delta.content,
        tool_calls: tail.tool_calls,
        malformed_tool_call: tail.malformed_tool_call,
        truncated_tool_call: tail.truncated_tool_call,
    }
}

fn retained_marker_prefix(text: &str, marker: &str) -> usize {
    (1..marker.len())
        .rev()
        .find(|&length| text.ends_with(&marker[..length]))
        .unwrap_or(0)
}

fn append_delta(destination: &mut AssistantDelta, source: AssistantDelta) {
    destination.reasoning.push_str(&source.reasoning);
    destination.content.push_str(&source.content);
}

fn parse_qwen_tool_calls(
    text: &str,
    contract: Option<&ToolCallConstraintSpec>,
) -> Option<(String, Vec<ParsedToolCall>)> {
    const CALL_CLOSE: &str = "</tool_call>";
    const FUNCTION_OPEN: &str = "<function=";
    const FUNCTION_CLOSE: &str = "</function>";
    const PARAMETER_OPEN: &str = "<parameter=";
    const PARAMETER_CLOSE: &str = "</parameter>";

    let first_call = text.find(CALL_OPEN)?;
    let content = text[..first_call].trim_end().to_owned();
    let mut remaining = &text[first_call..];
    let mut calls = Vec::new();

    loop {
        remaining = remaining.trim_start();
        if remaining.is_empty() {
            break;
        }
        remaining = remaining.strip_prefix(CALL_OPEN)?;
        remaining = remaining.trim_start();
        remaining = remaining.strip_prefix(FUNCTION_OPEN)?;
        let name_end = remaining.find('>')?;
        let name = remaining[..name_end].trim();
        if name.is_empty() || name.contains('<') {
            return None;
        }
        let contract_tool = match contract {
            Some(contract) => Some(contract.tool(name)?),
            None => None,
        };
        remaining = &remaining[name_end + 1..];
        let function_end = remaining.find(FUNCTION_CLOSE)?;
        let mut parameters = &remaining[..function_end];
        let mut arguments = Map::new();
        while !parameters.trim().is_empty() {
            parameters = parameters.trim_start();
            parameters = parameters.strip_prefix(PARAMETER_OPEN)?;
            let parameter_name_end = parameters.find('>')?;
            let parameter_name = parameters[..parameter_name_end].trim();
            if parameter_name.is_empty()
                || parameter_name.contains('<')
                || arguments.contains_key(parameter_name)
            {
                return None;
            }
            if contract_tool.is_some_and(|tool| tool.parameter(parameter_name).is_none()) {
                return None;
            }
            parameters = &parameters[parameter_name_end + 1..];
            let parameter_end = parameters.find(PARAMETER_CLOSE)?;
            let raw_value = parameters[..parameter_end].trim();
            let value = match serde_json::from_str(raw_value) {
                Ok(value) => value,
                Err(error) => {
                    if raw_value.starts_with('[') || raw_value.starts_with('{') {
                        eprintln!(
                            "TuiskoLLM tool call `{name}` parameter `{parameter_name}`: {} bytes opening as JSON did not parse ({error}); passing it through as a string",
                            raw_value.len()
                        );
                    }
                    Value::String(raw_value.to_owned())
                }
            };
            arguments.insert(parameter_name.to_owned(), value);
            parameters = &parameters[parameter_end + PARAMETER_CLOSE.len()..];
        }
        if contract_tool.is_some_and(|tool| {
            tool.parameters()
                .iter()
                .any(|parameter| parameter.required() && !arguments.contains_key(parameter.name()))
        }) {
            return None;
        }
        remaining = &remaining[function_end + FUNCTION_CLOSE.len()..];
        remaining = remaining.trim_start();
        remaining = remaining.strip_prefix(CALL_CLOSE)?;
        calls.push(ParsedToolCall {
            name: name.to_owned(),
            arguments: Value::Object(arguments).to_string(),
        });
    }

    (!calls.is_empty()).then_some((content, calls))
}

#[cfg(test)]
mod tests {
    use super::{
        AssistantStreamParser, parse_assistant_output, parse_assistant_output_constrained,
    };
    use serde_json::Value;
    use std::sync::Arc;
    use tuisko_frontend::{ToolCallConstraintSpec, ToolConstraintSpec, ToolParameterSpec};

    fn bash_constraint() -> Arc<ToolCallConstraintSpec> {
        Arc::new(
            ToolCallConstraintSpec::new(vec![
                ToolConstraintSpec::new(
                    "bash".into(),
                    vec![ToolParameterSpec::new("command".into(), true).unwrap()],
                )
                .unwrap(),
            ])
            .unwrap(),
        )
    }

    #[test]
    fn reasoning_parser_handles_split_end_tags() {
        let mut parser = AssistantStreamParser::new(true, false);
        let first = parser.push("reasoning</thi");
        let second = parser.push("nk>\n\nPa");
        let third = parser.push("ris");
        let tail = parser.finish();

        assert_eq!(first.reasoning, "reasoning");
        assert_eq!(second.content, "Pa");
        assert_eq!(third.content, "ris");
        assert_eq!(tail.delta, Default::default());
    }

    #[test]
    fn tool_mode_handles_a_marker_split_across_stream_deltas() {
        let mut parser = AssistantStreamParser::new(false, true);
        let first = parser.push("I will inspect it.<tool_");
        let second = parser
            .push("call><function=bash><parameter=command>ls</parameter></function></tool_call>");
        let tail = parser.finish();

        assert_eq!(first.content, "I will inspect it.");
        assert_eq!(second, Default::default());
        assert_eq!(tail.tool_calls.len(), 1);
        assert_eq!(tail.tool_calls[0].name, "bash");
        assert_eq!(tail.tool_calls[0].arguments, r#"{"command":"ls"}"#);
    }

    #[test]
    fn malformed_tool_xml_falls_back_to_plain_content() {
        let text = "<tool_call><function=bash><parameter=command>ls</function></tool_call>";
        let parsed = parse_assistant_output(text, false, true);

        assert_eq!(parsed.content, text);
        assert!(parsed.tool_calls.is_empty());
    }

    #[test]
    fn constrained_parser_rejects_the_observed_transposition() {
        let text = "<tool_call><function=command>\n</parameter></function></tool_call>";
        let parsed = parse_assistant_output_constrained(text, false, bash_constraint(), false);

        assert!(parsed.malformed_tool_call);
        assert!(parsed.content.is_empty());
        assert!(parsed.tool_calls.is_empty());
    }

    #[test]
    fn constrained_parser_discards_a_length_truncated_span() {
        let text = "prefix<tool_call><function=bash><parameter=command>ls";
        let parsed = parse_assistant_output_constrained(text, false, bash_constraint(), true);

        assert!(!parsed.malformed_tool_call);
        assert!(parsed.truncated_tool_call);
        assert_eq!(parsed.content, "prefix");
        assert!(parsed.tool_calls.is_empty());
    }

    #[test]
    fn constrained_parser_does_not_hide_a_complete_invalid_call_at_length() {
        let text = "<tool_call><function=bash></function></tool_call>";
        let parsed = parse_assistant_output_constrained(text, false, bash_constraint(), true);

        assert!(parsed.malformed_tool_call);
        assert!(!parsed.truncated_tool_call);
        assert!(parsed.content.is_empty());
        assert!(parsed.tool_calls.is_empty());
    }

    #[test]
    fn constrained_parser_requires_declared_parameters() {
        let parsed = parse_assistant_output_constrained(
            "<tool_call><function=bash></function></tool_call>",
            false,
            bash_constraint(),
            false,
        );

        assert!(parsed.malformed_tool_call);
        assert!(parsed.tool_calls.is_empty());
    }

    #[test]
    fn parses_multiple_typed_tool_parameters() {
        let parsed = parse_assistant_output(
            "inspect</think>\n\n<tool_call><function=edit><parameter=count>3</parameter><parameter=edits>[{\"newText\":\"x\"}]</parameter></function></tool_call>",
            true,
            true,
        );

        assert_eq!(parsed.reasoning, "inspect");
        assert_eq!(parsed.content, "");
        assert_eq!(parsed.tool_calls.len(), 1);
        let arguments: Value = serde_json::from_str(&parsed.tool_calls[0].arguments).unwrap();
        assert_eq!(arguments["count"], 3);
        assert!(arguments["edits"].is_array());
    }

    #[test]
    fn malformed_structured_parameter_remains_attributable_as_a_string() {
        let parsed = parse_assistant_output(
            "<tool_call><function=edit><parameter=edits>[{\"newText\":\"x\"</parameter></function></tool_call>",
            false,
            true,
        );
        let arguments: Value = serde_json::from_str(&parsed.tool_calls[0].arguments).unwrap();

        assert!(arguments["edits"].is_string());
    }

    #[test]
    fn every_utf8_split_matches_complete_assistant_parsing() {
        let text = "考える</think>\n\nI will inspect 🚀.<tool_call><function=bash><parameter=command>ls -la</parameter></function></tool_call>";
        let expected = parse_assistant_output(text, true, true);
        let boundaries = text
            .char_indices()
            .map(|(index, _)| index)
            .chain(std::iter::once(text.len()));

        for split in boundaries {
            let mut parser = AssistantStreamParser::new(true, true);
            let first = parser.push(&text[..split]);
            let second = parser.push(&text[split..]);
            let tail = parser.finish();
            let reasoning = first.reasoning + &second.reasoning + &tail.delta.reasoning;
            let content = first.content + &second.content + &tail.delta.content;

            assert_eq!(reasoning, expected.reasoning, "reasoning split at {split}");
            assert_eq!(content, expected.content, "content split at {split}");
            assert_eq!(
                tail.tool_calls, expected.tool_calls,
                "tool calls split at {split}"
            );
        }
    }
}
