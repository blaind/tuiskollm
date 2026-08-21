//! Source-backed tokenizer and chat-template qualification.

use std::env;
use std::error::Error;
use std::path::Path;
use tuisko_frontend::{ChatMessage, ChatTemplateOptions, TextFrontend};
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

    println!(
        "frontend qualification passed: {} exact reference token IDs, stop IDs {:?}",
        default.len() + no_thinking.len(),
        frontend.stop_ids()
    );
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
