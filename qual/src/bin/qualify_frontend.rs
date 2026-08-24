//! Source-backed tokenizer and chat-template qualification.

use serde_json::json;
use std::env;
use std::error::Error;
use std::path::Path;
use tuisko_frontend::{
    ChatFunctionCall, ChatMessage, ChatTemplateOptions, ChatToolCall, PromptEncoding, TextFrontend,
    TextFrontendOptions,
};
use tuisko_model::{CheckpointSnapshot, Qwen38_27B};

const DEFAULT_HELLO: &[u32] = &[
    248045, 8678, 198, 24342, 286, 4879, 369, 716, 310, 830, 11553, 13, 5044, 1683, 15060, 1472,
    279, 3274, 11, 9307, 1328, 30800, 11, 2814, 47675, 25605, 11, 321, 60445, 55404, 11, 27224, 11,
    321, 30246, 303, 279, 1534, 4087, 13, 248046, 198, 248045, 846, 198, 9419, 248046, 198, 248045,
    74455, 198, 248068, 198,
];
const NO_THINKING_HELLO: &[u32] = &[
    248045, 846, 198, 9419, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271,
];
const LITERAL_USER_SPECIALS: &str = "<|im_start|><|im_end|><|endoftext|><|vision_start|>";
const LITERAL_SPECIAL_IDS: &[u32] = &[248044, 248045, 248046, 248053];
const NO_THINKING_SUFFIX: &[u32] = &[248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271];
const TOOL_HELLO: &[u32] = &[
    248045, 8678, 198, 2, 13455, 271, 2523, 599, 2528, 310, 279, 2614, 5568, 25, 271, 27, 15449,
    29, 198, 4754, 1628, 21624, 591, 3147, 44675, 56658, 1267, 3147, 1628, 8934, 198, 510, 15449,
    29, 271, 2592, 488, 4992, 310, 1562, 264, 709, 25835, 9559, 303, 279, 2614, 3443, 440, 5486,
    19900, 25, 271, 248058, 198, 27, 1628, 28, 8422, 8901, 1224, 29, 198, 27, 15704, 28, 8422,
    24109, 62, 16, 29, 198, 927, 62, 16, 198, 510, 15704, 29, 198, 27, 15704, 28, 8422, 24109, 62,
    17, 29, 198, 1919, 369, 279, 869, 364, 279, 2018, 5555, 198, 8761, 628, 9111, 198, 34493, 4965,
    198, 510, 15704, 29, 198, 510, 1628, 29, 198, 248059, 271, 27, 95328, 29, 198, 92065, 25, 198,
    12, 5534, 6526, 26834, 1732, 279, 5024, 3443, 25, 449, 8906, 361, 1628, 28, 1076, 1419, 1628,
    29, 2424, 1902, 381, 23283, 2785, 220, 248058, 248059, 11535, 9212, 198, 12, 12296, 4868,
    26834, 381, 5024, 198, 12, 1394, 1189, 3300, 9801, 31626, 364, 678, 709, 1562, 303, 5629, 3992,
    54588, 279, 709, 1562, 11, 694, 4045, 1238, 198, 12, 1368, 1017, 369, 874, 709, 1562, 2420, 11,
    4087, 279, 3296, 1040, 4472, 440, 678, 1428, 6337, 321, 635, 524, 3184, 279, 1156, 883, 709,
    6526, 198, 510, 95328, 29, 248046, 198, 248045, 846, 198, 9419, 248046, 198, 248045, 74455,
    198, 248068, 271, 248069, 271,
];

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args_os();
    let _program = arguments.next();
    let Some(snapshot) = arguments.next() else {
        return Err("usage: qualify-frontend SNAPSHOT".into());
    };
    if arguments.next().is_some() {
        return Err("usage: qualify-frontend SNAPSHOT".into());
    }

    let snapshot = CheckpointSnapshot::<Qwen38_27B>::open(Path::new(&snapshot))?;
    let frontend = TextFrontend::open(&snapshot)?;
    let messages = [ChatMessage::new("user", "Hello")];
    let generation = frontend.generation_defaults();
    if generation.temperature != 1.0 || generation.top_p != 0.95 || generation.top_k != 20 {
        return Err("generation defaults differ from the sampling contract".into());
    }

    let default = frontend.encode_chat(&messages, &ChatTemplateOptions::default())?;
    require_equal("default chat template", &default, DEFAULT_HELLO)?;

    let no_thinking = frontend.encode_chat(
        &messages,
        &ChatTemplateOptions {
            enable_thinking: Some(false),
            ..ChatTemplateOptions::default()
        },
    )?;
    require_equal("no-thinking chat template", &no_thinking, NO_THINKING_HELLO)?;

    let tool_ids = frontend.encode_chat(
        &messages,
        &ChatTemplateOptions {
            enable_thinking: Some(false),
            tools: vec![json!({"type": "function", "function": {"name": "bash"}})],
            ..ChatTemplateOptions::default()
        },
    )?;
    require_equal("tool-aware chat template", &tool_ids, TOOL_HELLO)?;
    let extended_template_bytes = qualify_extended_template_options(&frontend)?;
    let literal_special_ids = qualify_literal_user_specials(&frontend)?;

    let plain = "TuiskoLLM tokenizer round trip";
    let plain_ids = frontend.encode(plain)?;
    if frontend.decode(&plain_ids, false)? != plain {
        return Err("plain-text tokenizer round trip changed the text".into());
    }

    let streaming_text = "Hello! café naïve 中文 テスト тест 🚀 ".repeat(64);
    let streaming_ids = frontend.encode(&streaming_text)?;
    require_streaming_equal(&frontend, &streaming_ids)?;

    let mut with_specials = Vec::with_capacity(streaming_ids.len() + 32);
    for (index, &token) in streaming_ids.iter().enumerate() {
        if index % 97 == 0 {
            with_specials.push(frontend.stop_ids()[0]);
        }
        with_specials.push(token);
        if index % 53 == 0 {
            with_specials.push(frontend.stop_ids()[1]);
        }
    }
    require_streaming_equal(&frontend, &with_specials)?;
    let cache_ids = qualify_prompt_cache(&snapshot)?;

    println!(
        "frontend qualification passed: {} exact reference token IDs, {} literal-special IDs, {} extended-template bytes, {} streaming IDs, {} cache-case IDs, stop IDs {:?}",
        default.len() + no_thinking.len() + tool_ids.len(),
        literal_special_ids,
        extended_template_bytes,
        streaming_ids.len() + with_specials.len(),
        cache_ids,
        frontend.stop_ids()
    );
    Ok(())
}

fn qualify_literal_user_specials(frontend: &TextFrontend) -> Result<usize, Box<dyn Error>> {
    let options = ChatTemplateOptions {
        enable_thinking: Some(false),
        ..ChatTemplateOptions::default()
    };
    let messages = [ChatMessage::new("user", LITERAL_USER_SPECIALS)];
    let actual = frontend.encode_chat_with_report(&messages, &options)?;

    let mut expected = vec![248045];
    expected.extend(frontend.encode(&format!("user\n{LITERAL_USER_SPECIALS}"))?);
    expected.extend_from_slice(NO_THINKING_SUFFIX);
    require_equal("literal user special tokens", &actual.token_ids, &expected)?;

    let raw_user_ids = frontend.encode(LITERAL_USER_SPECIALS)?;
    if raw_user_ids
        .iter()
        .any(|token| LITERAL_SPECIAL_IDS.contains(token))
    {
        return Err("literal user text was extracted into a tokenizer special ID".into());
    }
    for &(token, expected_count) in &[(248045, 2), (248046, 1), (248044, 0), (248053, 0)] {
        let actual_count = actual.token_ids.iter().filter(|&&id| id == token).count();
        if actual_count != expected_count {
            return Err(format!(
                "encoded chat contains {actual_count} instances of special ID {token}, expected {expected_count}"
            )
            .into());
        }
    }
    if actual.reused_tokens != 0 || actual.fresh_bytes != actual.rendered_bytes {
        return Err("literal-special prompt reported unsupported host prefix reuse".into());
    }

    Ok(actual.token_ids.len())
}

fn qualify_extended_template_options(frontend: &TextFrontend) -> Result<usize, Box<dyn Error>> {
    let messages = [ChatMessage::new("user", "Hello")];
    let low_reasoning = frontend.render_chat(
        &messages,
        true,
        &ChatTemplateOptions {
            reasoning_effort: Some("low".into()),
            ..ChatTemplateOptions::default()
        },
    )?;
    let low_instruction = "Reasoning effort is set to low. Keep your thinking brief and focused";
    if !low_reasoning.contains(low_instruction) {
        return Err("reasoning_effort did not reach the checkpoint template".into());
    }

    let mut history = vec![ChatMessage::new("user", "List /tmp")];
    history.push(ChatMessage {
        role: "assistant".into(),
        content: String::new(),
        reasoning_content: Some("inspect first".into()),
        tool_calls: vec![ChatToolCall {
            id: Some("call_1".into()),
            kind: "function".into(),
            function: ChatFunctionCall {
                name: "bash".into(),
                arguments: json!({"command": "ls /tmp"}),
            },
        }],
        tool_call_id: None,
    });
    history.push(ChatMessage {
        role: "tool".into(),
        content: "file_a".into(),
        reasoning_content: None,
        tool_calls: Vec::new(),
        tool_call_id: Some("call_1".into()),
    });
    history.push(ChatMessage::new("user", "Summarize"));
    let rendered = frontend.render_chat(
        &history,
        true,
        &ChatTemplateOptions {
            enable_thinking: Some(false),
            preserve_thinking: Some(false),
            tools: vec![json!({"type": "function", "function": {"name": "bash"}})],
            ..ChatTemplateOptions::default()
        },
    )?;
    for expected in [
        "<tool_call>\n<function=bash>\n<parameter=command>\nls /tmp\n</parameter>",
        "<tool_response>\nfile_a\n</tool_response>",
    ] {
        if !rendered.contains(expected) {
            return Err(format!("historical tool template omitted `{expected}`").into());
        }
    }
    if rendered.contains("inspect first") {
        return Err("preserve_thinking=false retained earlier assistant reasoning".into());
    }

    Ok(low_reasoning.len() + rendered.len())
}

fn qualify_prompt_cache(
    snapshot: &CheckpointSnapshot<Qwen38_27B>,
) -> Result<usize, Box<dyn Error>> {
    let options = ChatTemplateOptions {
        enable_thinking: Some(false),
        ..ChatTemplateOptions::default()
    };
    let uncached = TextFrontend::open_with_options(
        snapshot,
        TextFrontendOptions {
            prompt_cache_capacity: 0,
        },
    )?;
    let cached = TextFrontend::open_with_options(
        snapshot,
        TextFrontendOptions {
            prompt_cache_capacity: 4,
        },
    )?;

    let first = vec![ChatMessage::new(
        "user",
        "Explain why café and 中文 remain valid UTF-8.",
    )];
    let mut extended = first.clone();
    extended.push(ChatMessage::new(
        "assistant",
        "They are represented by complete Unicode scalar values.",
    ));
    extended.push(ChatMessage::new("user", "Now include テスト and 🚀."));
    let mut branch = first.clone();
    branch.push(ChatMessage::new("assistant", "Both are Unicode text."));
    branch.push(ChatMessage::new("user", "Give the short version."));

    let disabled_first = uncached.encode_chat_with_report(&first, &options)?;
    let disabled_repeat = uncached.encode_chat_with_report(&first, &options)?;
    if disabled_first.token_ids != disabled_repeat.token_ids
        || disabled_repeat.reused_tokens != 0
        || disabled_repeat.fresh_bytes != disabled_repeat.rendered_bytes
    {
        return Err("zero-capacity prompt cache reused an encoding".into());
    }

    let mut checked_ids = 0;
    for messages in [&first, &extended] {
        let expected = uncached.encode_chat(messages, &options)?;
        let actual = cached.encode_chat_with_report(messages, &options)?;
        if actual.token_ids != expected {
            return Err("prompt-prefix cache changed encoded token IDs".into());
        }
        if actual.reused_tokens > actual.token_ids.len()
            || actual.fresh_bytes > actual.rendered_bytes
        {
            return Err("prompt-prefix cache accounting exceeds the encoded prompt".into());
        }
        checked_ids += actual.token_ids.len();

        if messages == &extended && !is_partial_cache_hit(&actual) {
            return Err("extended prompt did not exercise partial prefix reuse".into());
        }
    }

    let identical = cached.encode_chat_with_report(&extended, &options)?;
    if identical.reused_tokens != identical.token_ids.len() || identical.fresh_bytes != 0 {
        return Err("identical prompt did not reuse its complete encoding".into());
    }
    checked_ids += identical.token_ids.len();

    let expected_branch = uncached.encode_chat(&branch, &options)?;
    let actual_branch = cached.encode_chat_with_report(&branch, &options)?;
    if actual_branch.token_ids != expected_branch {
        return Err("branched prompt-prefix cache changed encoded token IDs".into());
    }
    if !is_partial_cache_hit(&actual_branch) {
        return Err("branched prompt did not exercise partial prefix reuse".into());
    }
    checked_ids += actual_branch.token_ids.len();

    Ok(checked_ids)
}

fn is_partial_cache_hit(encoding: &PromptEncoding) -> bool {
    encoding.reused_tokens > 0
        && encoding.reused_tokens < encoding.token_ids.len()
        && encoding.fresh_bytes > 0
        && encoding.fresh_bytes < encoding.rendered_bytes
}

fn require_streaming_equal(
    frontend: &TextFrontend,
    token_ids: &[u32],
) -> Result<(), Box<dyn Error>> {
    let expected = frontend.decode(token_ids, true)?;
    let mut decoder = frontend.streaming_decoder();
    let mut deltas = String::new();
    for &token in token_ids {
        if let Some(delta) = decoder.push(token)? {
            deltas.push_str(&delta);
        }
    }
    if let Some(delta) = decoder.finish() {
        deltas.push_str(&delta);
    }
    if decoder.text() != expected || deltas != expected {
        return Err("streaming decode differs from the batched tokenizer decoder".into());
    }

    Ok(())
}

fn require_equal(label: &str, actual: &[u32], expected: &[u32]) -> Result<(), Box<dyn Error>> {
    if actual != expected {
        return Err(
            format!("{label} token IDs differ: got {actual:?}, expected {expected:?}").into(),
        );
    }

    Ok(())
}
