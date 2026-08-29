//! Tokenizer-bound constraints for the pinned Qwen XML tool-call protocol.

use crate::{DecodeState, FrontendError, FrontendResult};
use std::collections::HashSet;
use std::sync::Arc;

const CALL_OPEN: &[u8] = b"<tool_call>";
const FUNCTION_CLOSE: &[u8] = b"</function>";
const PARAMETER_CLOSE: &[u8] = b"</parameter>";
const CALL_CLOSE: &[u8] = b"</tool_call>";
const MAX_TOOLS: usize = 64;
const MAX_PARAMETERS: usize = 64;
const MAX_NAME_BYTES: usize = 8 * 1024;

/// One declared parameter visible to the Qwen tool-call protocol.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ToolParameterSpec {
    name: String,
    required: bool,
}

impl ToolParameterSpec {
    /// Creates one checked declared parameter.
    pub fn new(name: String, required: bool) -> FrontendResult<Self> {
        require_protocol_name("tool parameter", &name)?;
        Ok(Self { name, required })
    }

    /// Declared parameter name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Whether the function must contain this parameter before it closes.
    pub const fn required(&self) -> bool {
        self.required
    }
}

/// One function and its finite structural parameter inventory.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ToolConstraintSpec {
    name: String,
    parameters: Vec<ToolParameterSpec>,
}

impl ToolConstraintSpec {
    /// Creates one checked function specification.
    pub fn new(name: String, parameters: Vec<ToolParameterSpec>) -> FrontendResult<Self> {
        require_protocol_name("tool function", &name)?;
        if parameters.len() > MAX_PARAMETERS {
            return Err(FrontendError::Contract(format!(
                "tool function `{name}` declares {} parameters; at most {MAX_PARAMETERS} are admitted",
                parameters.len()
            )));
        }
        let mut names = HashSet::with_capacity(parameters.len());
        for parameter in &parameters {
            if !names.insert(parameter.name.as_str()) {
                return Err(FrontendError::Contract(format!(
                    "tool function `{name}` repeats parameter `{}`",
                    parameter.name
                )));
            }
        }
        Ok(Self { name, parameters })
    }

    /// Function name exposed to the model.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Finite declared parameter inventory.
    pub fn parameters(&self) -> &[ToolParameterSpec] {
        &self.parameters
    }

    /// Finds one declared parameter by name.
    pub fn parameter(&self, name: &str) -> Option<&ToolParameterSpec> {
        self.parameters
            .iter()
            .find(|parameter| parameter.name == name)
    }
}

/// Immutable tool-call contract shared by generation and response validation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ToolCallConstraintSpec {
    tools: Vec<ToolConstraintSpec>,
}

impl ToolCallConstraintSpec {
    /// Creates a bounded, unique function inventory.
    pub fn new(tools: Vec<ToolConstraintSpec>) -> FrontendResult<Self> {
        if tools.is_empty() {
            return Err(FrontendError::Contract(
                "tool-call constraint requires at least one function".into(),
            ));
        }
        if tools.len() > MAX_TOOLS {
            return Err(FrontendError::Contract(format!(
                "tool-call constraint declares {} functions; at most {MAX_TOOLS} are admitted",
                tools.len()
            )));
        }
        let mut names = HashSet::with_capacity(tools.len());
        let mut name_bytes = 0usize;
        for tool in &tools {
            if !names.insert(tool.name.as_str()) {
                return Err(FrontendError::Contract(format!(
                    "tool-call constraint repeats function `{}`",
                    tool.name
                )));
            }
            name_bytes = name_bytes
                .checked_add(tool.name.len())
                .and_then(|total| {
                    tool.parameters.iter().try_fold(total, |sum, parameter| {
                        sum.checked_add(parameter.name.len())
                    })
                })
                .ok_or_else(|| FrontendError::Contract("tool name bytes overflow usize".into()))?;
        }
        if name_bytes > MAX_NAME_BYTES {
            return Err(FrontendError::Contract(format!(
                "tool-call constraint contains {name_bytes} function and parameter name bytes; at most {MAX_NAME_BYTES} are admitted"
            )));
        }
        Ok(Self { tools })
    }

    /// Admitted function inventory.
    pub fn tools(&self) -> &[ToolConstraintSpec] {
        &self.tools
    }

    /// Finds one admitted function by name.
    pub fn tool(&self, name: &str) -> Option<&ToolConstraintSpec> {
        self.tools.iter().find(|tool| tool.name == name)
    }
}

fn require_protocol_name(kind: &str, name: &str) -> FrontendResult<()> {
    if name.is_empty()
        || name.len() > 64
        || !name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
    {
        return Err(FrontendError::Contract(format!(
            "{kind} name must contain 1..=64 ASCII letters, digits, `_`, or `-`"
        )));
    }
    Ok(())
}

pub(crate) struct TokenByteInventory {
    entries: Vec<TokenByteSpan>,
    bytes: Vec<u8>,
}

#[derive(Clone, Copy)]
struct TokenByteSpan {
    start: u32,
    len: u16,
}

impl TokenByteInventory {
    fn build(decode: &DecodeState) -> FrontendResult<Self> {
        let vocabulary = decode.tokenizer.get_vocab_size(true);
        let mut entries = Vec::with_capacity(vocabulary);
        let mut bytes = Vec::new();
        for token_id in 0..vocabulary {
            let token_id = u32::try_from(token_id)
                .map_err(|_| FrontendError::Contract("tokenizer entry count exceeds u32".into()))?;
            if decode.special_decode_ids.contains(&token_id) {
                entries.push(TokenByteSpan {
                    start: 0,
                    len: u16::MAX,
                });
                continue;
            }
            let token = decode.tokenizer.id_to_token(token_id).ok_or_else(|| {
                FrontendError::Tokenizer(format!(
                    "token ID {token_id} has no tokenizer entry while building tool constraints"
                ))
            })?;
            let start = u32::try_from(bytes.len()).map_err(|_| {
                FrontendError::Contract("tool token-byte inventory exceeds 4 GiB".into())
            })?;
            for character in token.chars() {
                bytes.push(decode.byte_table.get(&character).copied().ok_or_else(|| {
                    FrontendError::Tokenizer(format!(
                        "token ID {token_id} contains non-byte-level character {character:?}"
                    ))
                })?);
            }
            let len = u16::try_from(bytes.len() - start as usize).map_err(|_| {
                FrontendError::Contract(format!(
                    "token ID {token_id} represents more than {} bytes",
                    u16::MAX - 1
                ))
            })?;
            if len == u16::MAX {
                return Err(FrontendError::Contract(format!(
                    "token ID {token_id} represents more than {} bytes",
                    u16::MAX - 1
                )));
            }
            entries.push(TokenByteSpan { start, len });
        }
        Ok(Self { entries, bytes })
    }

    fn bytes(&self, token_id: u32) -> Option<&[u8]> {
        let span = self.entries.get(token_id as usize)?;
        if span.len == u16::MAX {
            return None;
        }
        let start = span.start as usize;
        Some(&self.bytes[start..start + usize::from(span.len)])
    }
}

#[derive(Clone)]
struct CompiledParameter {
    open: Box<[u8]>,
}

#[derive(Clone)]
struct CompiledTool {
    function_open: Box<[u8]>,
    parameters: Vec<CompiledParameter>,
    required: u64,
}

#[derive(Clone)]
struct CompiledSpec {
    tools: Vec<CompiledTool>,
}

impl CompiledSpec {
    fn new(spec: &ToolCallConstraintSpec) -> Self {
        Self {
            tools: spec
                .tools
                .iter()
                .map(|tool| CompiledTool {
                    function_open: format!("<function={}>", tool.name)
                        .into_bytes()
                        .into_boxed_slice(),
                    parameters: tool
                        .parameters
                        .iter()
                        .map(|parameter| CompiledParameter {
                            open: format!("<parameter={}>", parameter.name)
                                .into_bytes()
                                .into_boxed_slice(),
                        })
                        .collect(),
                    required: tool
                        .parameters
                        .iter()
                        .enumerate()
                        .fold(0u64, |mask, (index, parameter)| {
                            mask | (u64::from(parameter.required) << index)
                        }),
                })
                .collect(),
        }
    }
}

const PREFIX_CAPACITY: usize = 80;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PrefixBuffer {
    bytes: [u8; PREFIX_CAPACITY],
    len: u8,
}

impl PrefixBuffer {
    const fn new() -> Self {
        Self {
            bytes: [0; PREFIX_CAPACITY],
            len: 0,
        }
    }

    const fn is_empty(self) -> bool {
        self.len == 0
    }

    fn as_slice(&self) -> &[u8] {
        &self.bytes[..usize::from(self.len)]
    }

    fn push(&mut self, byte: u8) -> bool {
        let index = usize::from(self.len);
        if index == PREFIX_CAPACITY {
            return false;
        }
        self.bytes[index] = byte;
        self.len += 1;
        true
    }

    fn clear(&mut self) {
        self.len = 0;
    }

    fn remove_first(&mut self) {
        let len = usize::from(self.len);
        self.bytes.copy_within(1..len, 0);
        self.len -= 1;
    }
}

/// Tokenizer-independent protocol state used for provisional MTP evaluation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ToolCallState(ProtocolState);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ProtocolState {
    Content {
        marker: PrefixBuffer,
    },
    BeforeFunction {
        prefix: PrefixBuffer,
    },
    BeforeParameter {
        tool: usize,
        used: u64,
        prefix: PrefixBuffer,
    },
    ParameterValue {
        tool: usize,
        parameter: usize,
        used: u64,
        marker: PrefixBuffer,
    },
    BeforeCallClose {
        prefix: PrefixBuffer,
    },
    AfterCall {
        prefix: PrefixBuffer,
    },
    UnreachableAfterStop,
}

impl Default for ToolCallState {
    fn default() -> Self {
        Self(ProtocolState::Content {
            marker: PrefixBuffer::new(),
        })
    }
}

/// One request's committed lazy Qwen tool-call constraint.
pub struct ToolCallConstraint {
    token_bytes: Arc<TokenByteInventory>,
    stop_ids: Vec<u32>,
    compiled: Arc<CompiledSpec>,
    committed: ToolCallState,
}

impl ToolCallConstraint {
    pub(crate) fn new(
        decode: Arc<DecodeState>,
        stop_ids: Vec<u32>,
        spec: Arc<ToolCallConstraintSpec>,
    ) -> FrontendResult<Self> {
        let token_bytes = {
            let mut cached = decode
                .tool_token_bytes
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if cached.is_none() {
                *cached = Some(Arc::new(TokenByteInventory::build(&decode)?));
            }
            cached
                .as_ref()
                .expect("tool token bytes were initialized")
                .clone()
        };
        Ok(Self {
            token_bytes,
            stop_ids,
            compiled: Arc::new(CompiledSpec::new(&spec)),
            committed: ToolCallState::default(),
        })
    }

    /// Returns committed state advanced through an uncommitted MTP prefix.
    pub fn provisional_state(&self, provisional: &[u32]) -> FrontendResult<ToolCallState> {
        let mut state = self.committed;
        for &token in provisional {
            self.advance_token(&mut state, token)?;
        }
        Ok(state)
    }

    /// Whether one token can atomically advance the supplied state.
    pub fn allows_token(&self, state: &ToolCallState, token: u32) -> bool {
        let mut candidate = *state;
        self.advance_token(&mut candidate, token).is_ok()
    }

    /// Advances committed state through one selected token.
    pub fn commit_token(&mut self, token: u32) -> FrontendResult<()> {
        let mut state = self.committed;
        self.advance_token(&mut state, token)?;
        self.committed = state;
        Ok(())
    }

    /// Whether generation has entered but not completed a tool-call span.
    pub fn has_incomplete_call(&self) -> bool {
        !matches!(
            self.committed.0,
            ProtocolState::Content { .. }
                | ProtocolState::AfterCall { .. }
                | ProtocolState::UnreachableAfterStop
        )
    }

    fn advance_token(&self, state: &mut ToolCallState, token: u32) -> FrontendResult<()> {
        let stopped = self.stop_ids.contains(&token);
        if stopped {
            return match state.0 {
                ProtocolState::Content { .. }
                | ProtocolState::AfterCall { .. }
                | ProtocolState::UnreachableAfterStop => {
                    state.0 = ProtocolState::UnreachableAfterStop;
                    Ok(())
                }
                _ => Err(FrontendError::Contract(
                    "tool call cannot stop before its closing tag".into(),
                )),
            };
        }
        if matches!(state.0, ProtocolState::UnreachableAfterStop) {
            return Ok(());
        }
        let Some(bytes) = self.token_bytes.bytes(token) else {
            return if matches!(state.0, ProtocolState::Content { .. }) {
                Ok(())
            } else {
                Err(FrontendError::Contract(format!(
                    "special or padded token {token} is not admitted inside a tool call"
                )))
            };
        };
        for &byte in bytes {
            if !advance_byte(&mut state.0, byte, &self.compiled) {
                return Err(FrontendError::Contract(format!(
                    "token {token} violates the active Qwen tool-call protocol"
                )));
            }
        }
        Ok(())
    }
}

fn advance_byte(state: &mut ProtocolState, byte: u8, spec: &CompiledSpec) -> bool {
    match state {
        ProtocolState::Content { marker } => {
            if advance_marker(marker, byte, &[CALL_OPEN]) == Some(0) {
                *state = ProtocolState::BeforeFunction {
                    prefix: PrefixBuffer::new(),
                };
            }
            true
        }
        ProtocolState::BeforeFunction { prefix } => {
            if prefix.is_empty() && byte.is_ascii_whitespace() {
                return true;
            }
            if !prefix.push(byte) {
                return false;
            }
            match match_choice(
                prefix.as_slice(),
                spec.tools.iter().map(|tool| tool.function_open.as_ref()),
            ) {
                ChoiceMatch::Prefix => true,
                ChoiceMatch::Complete(tool) => {
                    *state = ProtocolState::BeforeParameter {
                        tool,
                        used: 0,
                        prefix: PrefixBuffer::new(),
                    };
                    true
                }
                ChoiceMatch::Invalid => false,
            }
        }
        ProtocolState::BeforeParameter { tool, used, prefix } => {
            if prefix.is_empty() && byte.is_ascii_whitespace() {
                return true;
            }
            if !prefix.push(byte) {
                return false;
            }
            let selected_tool = &spec.tools[*tool];
            let required_left = selected_tool.required & !*used;
            if required_left == 0 && FUNCTION_CLOSE.starts_with(prefix.as_slice()) {
                if prefix.as_slice() == FUNCTION_CLOSE {
                    *state = ProtocolState::BeforeCallClose {
                        prefix: PrefixBuffer::new(),
                    };
                }
                return true;
            }
            let mut any_prefix = false;
            let mut complete = None;
            for (parameter, compiled) in selected_tool.parameters.iter().enumerate() {
                if *used & (1u64 << parameter) != 0 {
                    continue;
                }
                if compiled.open.starts_with(prefix.as_slice()) {
                    any_prefix = true;
                    if compiled.open.as_ref() == prefix.as_slice() {
                        complete = Some(parameter);
                        break;
                    }
                }
            }
            if let Some(parameter) = complete {
                *state = ProtocolState::ParameterValue {
                    tool: *tool,
                    parameter,
                    used: *used,
                    marker: PrefixBuffer::new(),
                };
                true
            } else {
                any_prefix
            }
        }
        ProtocolState::ParameterValue {
            tool,
            parameter,
            used,
            marker,
        } => match advance_marker(marker, byte, &[PARAMETER_CLOSE, FUNCTION_CLOSE, CALL_CLOSE]) {
            Some(0) => {
                *state = ProtocolState::BeforeParameter {
                    tool: *tool,
                    used: *used | (1u64 << *parameter),
                    prefix: PrefixBuffer::new(),
                };
                true
            }
            Some(_) => false,
            None => true,
        },
        ProtocolState::BeforeCallClose { prefix } => {
            if prefix.is_empty() && byte.is_ascii_whitespace() {
                return true;
            }
            if !prefix.push(byte) {
                return false;
            }
            if !CALL_CLOSE.starts_with(prefix.as_slice()) {
                return false;
            }
            if prefix.as_slice() == CALL_CLOSE {
                *state = ProtocolState::AfterCall {
                    prefix: PrefixBuffer::new(),
                };
            }
            true
        }
        ProtocolState::AfterCall { prefix } => {
            if prefix.is_empty() && byte.is_ascii_whitespace() {
                return true;
            }
            if !prefix.push(byte) {
                return false;
            }
            if !CALL_OPEN.starts_with(prefix.as_slice()) {
                return false;
            }
            if prefix.as_slice() == CALL_OPEN {
                *state = ProtocolState::BeforeFunction {
                    prefix: PrefixBuffer::new(),
                };
            }
            true
        }
        ProtocolState::UnreachableAfterStop => true,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ChoiceMatch {
    Prefix,
    Complete(usize),
    Invalid,
}

fn match_choice<'a>(prefix: &[u8], options: impl Iterator<Item = &'a [u8]>) -> ChoiceMatch {
    let mut any = false;
    for (index, option) in options.enumerate() {
        if option.starts_with(prefix) {
            any = true;
            if option == prefix {
                return ChoiceMatch::Complete(index);
            }
        }
    }
    if any {
        ChoiceMatch::Prefix
    } else {
        ChoiceMatch::Invalid
    }
}

fn advance_marker(marker: &mut PrefixBuffer, byte: u8, patterns: &[&[u8]]) -> Option<usize> {
    if !marker.push(byte) {
        marker.clear();
        return None;
    }
    if let Some(index) = patterns
        .iter()
        .position(|pattern| *pattern == marker.as_slice())
    {
        marker.clear();
        return Some(index);
    }
    while !marker.is_empty()
        && !patterns
            .iter()
            .any(|pattern| pattern.starts_with(marker.as_slice()))
    {
        marker.remove_first();
    }
    None
}

#[cfg(test)]
mod tests {
    use super::{
        CALL_OPEN, CompiledSpec, PrefixBuffer, ProtocolState, ToolCallConstraintSpec,
        ToolConstraintSpec, ToolParameterSpec, advance_byte,
    };

    fn spec() -> CompiledSpec {
        let spec = ToolCallConstraintSpec::new(vec![
            ToolConstraintSpec::new(
                "bash".into(),
                vec![ToolParameterSpec::new("command".into(), true).unwrap()],
            )
            .unwrap(),
            ToolConstraintSpec::new(
                "read".into(),
                vec![ToolParameterSpec::new("path".into(), true).unwrap()],
            )
            .unwrap(),
        ])
        .unwrap();
        CompiledSpec::new(&spec)
    }

    fn accepts(text: &str) -> bool {
        let spec = spec();
        let mut state = ProtocolState::Content {
            marker: PrefixBuffer::new(),
        };
        text.bytes()
            .all(|byte| advance_byte(&mut state, byte, &spec))
    }

    #[test]
    fn accepts_complete_registered_call() {
        assert!(accepts(
            "prefix<tool_call>\n<function=bash>\n<parameter=command>ls\n</parameter>\n</function>\n</tool_call>"
        ));
    }

    #[test]
    fn rejects_observed_function_parameter_transposition() {
        assert!(!accepts(
            "<tool_call>\n<function=command>\n\n</parameter>\n</function>\n</tool_call>"
        ));
    }

    #[test]
    fn requires_declared_parameter_before_function_close() {
        assert!(!accepts("<tool_call><function=bash></function>"));
    }

    #[test]
    fn rejects_unknown_and_duplicate_parameters() {
        assert!(!accepts(
            "<tool_call><function=bash><parameter=path>x</parameter>"
        ));
        assert!(!accepts(
            "<tool_call><function=bash><parameter=command>x</parameter><parameter=command>y</parameter>"
        ));
    }

    #[test]
    fn rejects_premature_structural_closes_inside_a_value() {
        assert!(!accepts(
            "<tool_call><function=bash><parameter=command>x</function>"
        ));
        assert!(!accepts(
            "<tool_call><function=bash><parameter=command>x</tool_call>"
        ));
    }

    #[test]
    fn accepts_multiple_complete_calls() {
        assert!(accepts(
            "<tool_call><function=bash><parameter=command>ls</parameter></function></tool_call>\n<tool_call><function=read><parameter=path>/tmp/x</parameter></function></tool_call>"
        ));
    }

    #[test]
    fn trigger_overlap_and_one_byte_steps_work() {
        let spec = spec();
        let mut state = ProtocolState::Content {
            marker: PrefixBuffer::new(),
        };
        for byte in
            b"<tool_<tool_call><function=read><parameter=path>x</parameter></function></tool_call>"
        {
            assert!(advance_byte(&mut state, *byte, &spec));
        }
        assert!(matches!(state, ProtocolState::AfterCall { .. }));
    }

    #[test]
    fn call_trigger_is_the_expected_pinned_literal() {
        assert_eq!(CALL_OPEN, b"<tool_call>");
    }
}
