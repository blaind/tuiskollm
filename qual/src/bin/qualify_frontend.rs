//! Source-backed tokenizer and chat-template qualification.

use std::env;
use std::error::Error;
use std::path::Path;
use tuisko_frontend::{
    ChatMessage, ChatTemplateOptions, PromptEncoding, TextFrontend, TextFrontendOptions,
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

    let default = frontend.encode_chat(&messages, ChatTemplateOptions::default())?;
    require_equal("default chat template", &default, DEFAULT_HELLO)?;

    let no_thinking = frontend.encode_chat(
        &messages,
        ChatTemplateOptions {
            enable_thinking: Some(false),
        },
    )?;
    require_equal("no-thinking chat template", &no_thinking, NO_THINKING_HELLO)?;

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
        "frontend qualification passed: {} exact reference token IDs, {} streaming IDs, {} cache-case IDs, stop IDs {:?}",
        default.len() + no_thinking.len(),
        streaming_ids.len() + with_specials.len(),
        cache_ids,
        frontend.stop_ids()
    );
    Ok(())
}

fn qualify_prompt_cache(
    snapshot: &CheckpointSnapshot<Qwen38_27B>,
) -> Result<usize, Box<dyn Error>> {
    let options = ChatTemplateOptions {
        enable_thinking: Some(false),
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

    let disabled_first = uncached.encode_chat_with_report(&first, options)?;
    let disabled_repeat = uncached.encode_chat_with_report(&first, options)?;
    if disabled_first.token_ids != disabled_repeat.token_ids
        || disabled_repeat.reused_tokens != 0
        || disabled_repeat.fresh_bytes != disabled_repeat.rendered_bytes
    {
        return Err("zero-capacity prompt cache reused an encoding".into());
    }

    let mut checked_ids = 0;
    for messages in [&first, &extended] {
        let expected = uncached.encode_chat(messages, options)?;
        let actual = cached.encode_chat_with_report(messages, options)?;
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

    let identical = cached.encode_chat_with_report(&extended, options)?;
    if identical.reused_tokens != identical.token_ids.len() || identical.fresh_bytes != 0 {
        return Err("identical prompt did not reuse its complete encoding".into());
    }
    checked_ids += identical.token_ids.len();

    let expected_branch = uncached.encode_chat(&branch, options)?;
    let actual_branch = cached.encode_chat_with_report(&branch, options)?;
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
