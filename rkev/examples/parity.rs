//! Compare the local engine's answers against a Kev server's, question by
//! question. This is the check that decides whether the engine is right: every
//! other test in this crate compares the implementation with itself or with a
//! transcription of the reference, never with the reference running.
//!
//! The server side is a recorded response, which is what makes the recording a
//! fixture a later run can use.
//!
//! ```bash
//! # 1. a server, on the exact path the published numbers use
//! KEV_DTYPE=fp32 uv run --extra serve python -m kev.serve \
//!     --run jaredpalmer/kev-4b@qwen3 --port 8009
//!
//! # 2. the request, and what the server answers
//! curl -s localhost:8009/v1/systemone -H 'content-type: application/json' \
//!     -d @request.json > server.json
//!
//! # 3. the same request, in process
//! cargo run --features candle --example parity -- \
//!     --base   ~/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Base/snapshots/<rev> \
//!     --checkpoint ~/.cache/huggingface/hub/models--jaredpalmer--kev-4b/snapshots/<rev> \
//!     --request request.json --server server.json
//! ```
//!
//! Exits non-zero when a probability differs by more than `--tolerance`.

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::process::ExitCode;

use rkev::{pointer_head, Answer, Backend, LocalEngine, SystemOneRequest, SystemOneResponse};

struct Options {
    base: PathBuf,
    checkpoint: Option<PathBuf>,
    head: Option<PathBuf>,
    request: PathBuf,
    server: PathBuf,
    tolerance: f64,
    /// The recording came from `/v1/systemone/separate`, so answer it that way.
    separate: bool,
}

fn main() -> ExitCode {
    let options = match parse() {
        Ok(options) => options,
        Err(message) => {
            eprintln!("{message}\n\n{USAGE}");
            return ExitCode::FAILURE;
        }
    };
    match run(&options) {
        Ok(Verdict { worst, same_prompt }) if worst <= options.tolerance && same_prompt => {
            println!(
                "\nlargest difference {worst:.5}, within {}",
                options.tolerance
            );
            ExitCode::SUCCESS
        }
        Ok(Verdict { worst, same_prompt }) => {
            if !same_prompt {
                println!(
                    "\nthe two sides read a different number of input tokens, so they \
                     answered different prompts; agreeing probabilities would be luck"
                );
            }
            // Not "over the tolerance" when it was the prompt that failed: the
            // difference may well be zero, and saying otherwise sends the reader
            // after the wrong thing.
            println!(
                "\nlargest difference {worst:.5}, {} {}",
                if worst <= options.tolerance {
                    "within"
                } else {
                    "over"
                },
                options.tolerance
            );
            ExitCode::FAILURE
        }
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

/// What the comparison found.
struct Verdict {
    /// The largest difference between two probabilities for the same option.
    worst: f64,
    /// Whether both sides tokenised the same prompt. `input_tokens` is the only
    /// evidence of that available here, and it is decisive: probabilities can
    /// agree to four decimals over prompts that differ, these cannot.
    same_prompt: bool,
}

fn run(options: &Options) -> Result<Verdict, Box<dyn std::error::Error>> {
    let request: SystemOneRequest =
        serde_json::from_str(&std::fs::read_to_string(&options.request)?)?;
    let expected: SystemOneResponse =
        serde_json::from_str(&std::fs::read_to_string(&options.server)?)?;

    let checkpoint = options.checkpoint.as_deref();
    let head = options.head.clone().unwrap_or_else(|| {
        let dir = checkpoint.unwrap_or(&options.base);
        [dir.join("head.pt"), dir.join("head.safetensors")]
            .into_iter()
            .find(|path| path.exists())
            .unwrap_or_else(|| dir.join("head.pt"))
    });
    let mut engine = LocalEngine::new(
        Backend::open(&options.base, checkpoint)?,
        pointer_head(&head)?,
    );
    // Serving the layout the checkpoint was not trained on is a silently
    // different prompt, which would show up here as a difference that is ours.
    if let Some(isolation) = rkev::option_isolation(&head)? {
        engine = engine.with_option_isolation(isolation);
    }

    let answered = if options.separate {
        engine.system_one_separate_blocking(&request)?
    } else {
        engine.system_one_blocking(&request)?
    };

    if expected.model != answered.model {
        println!(
            "note             the server answered as {:?}, the engine as {:?}",
            expected.model, answered.model
        );
    }
    // A recording may predate `usage`, so zero means "not recorded" rather than
    // "differs". output_tokens counts the serialised answers, so it moves with
    // any difference in the probabilities and is a note, not a verdict.
    let server_tokens = expected.usage.input_tokens;
    let same_prompt = server_tokens == 0 || server_tokens == answered.usage.input_tokens;
    if !same_prompt {
        println!(
            "input_tokens     {:<14} {server_tokens:>9} {:>9}   DIFFERENT",
            "", answered.usage.input_tokens
        );
    }
    if expected.usage.output_tokens != 0
        && expected.usage.output_tokens != answered.usage.output_tokens
    {
        println!(
            "note             output_tokens {} against {}, which follows the answers",
            expected.usage.output_tokens, answered.usage.output_tokens
        );
    }

    println!(
        "{:<16} {:<14} {:>9} {:>9} {:>9}",
        "question", "option", "server", "local", "diff"
    );

    let mut worst = 0.0f64;
    let ids: BTreeSet<&String> = expected
        .answers
        .keys()
        .chain(answered.answers.keys())
        .collect();
    for id in ids {
        let (Some(expected), Some(local)) = (expected.answer(id), answered.answer(id)) else {
            println!("{id:<16} MISSING on one side");
            return Ok(Verdict {
                worst: f64::INFINITY,
                same_prompt,
            });
        };
        for (option, expected, local) in pairs(expected, local) {
            let difference = (expected - local).abs();
            worst = worst.max(difference);
            println!("{id:<16} {option:<14} {expected:>9.4} {local:>9.4} {difference:>9.5}");
        }
    }
    Ok(Verdict { worst, same_prompt })
}

/// Every probability of one answer, by name, from both sides.
fn pairs(expected: &Answer, local: &Answer) -> Vec<(String, f64, f64)> {
    match (expected.probabilities(), local.probabilities()) {
        (Some(expected), Some(local)) => expected
            .iter()
            .map(|(name, value)| {
                (
                    name.clone(),
                    *value,
                    local.get(name).copied().unwrap_or(f64::NAN),
                )
            })
            .collect(),
        // A yes/no answer reports no distribution; the probability is the answer.
        _ => vec![(
            String::from("yes"),
            expected.as_noul().unwrap_or(f64::NAN),
            local.as_noul().unwrap_or(f64::NAN),
        )],
    }
}

const USAGE: &str = "\
usage: parity --base <dir> --request <file> --server <file>
              [--checkpoint <dir>] [--head <file>] [--tolerance <f64>]

  --base        the base model directory (config.json and its safetensors)
  --checkpoint  the Kev checkpoint: adapter, head.pt, tokenizer
  --head        the pointer head, if it is not <checkpoint>/head.pt
  --separate    the recording is of /v1/systemone/separate
  --request     the request body, the same one the server was sent
  --server      the server's response
  --tolerance   largest probability difference to accept (default 0.01)";

fn parse() -> Result<Options, String> {
    let mut base = None;
    let mut checkpoint = None;
    let mut head = None;
    let mut request = None;
    let mut server = None;
    let mut tolerance = 0.01;
    let mut separate = false;

    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        let mut value = || {
            arguments
                .next()
                .ok_or_else(|| format!("{argument} needs a value"))
        };
        match argument.as_str() {
            "--base" => base = Some(PathBuf::from(value()?)),
            "--checkpoint" => checkpoint = Some(PathBuf::from(value()?)),
            "--head" => head = Some(PathBuf::from(value()?)),
            "--request" => request = Some(PathBuf::from(value()?)),
            "--server" => server = Some(PathBuf::from(value()?)),
            "--tolerance" => {
                tolerance = value()?.parse().map_err(|e| format!("--tolerance: {e}"))?
            }
            "--separate" => separate = true,
            "-h" | "--help" => return Err(String::from("parity")),
            other => return Err(format!("unknown argument {other}")),
        }
    }

    Ok(Options {
        base: base.ok_or("--base is required")?,
        checkpoint,
        head,
        request: request.ok_or("--request is required")?,
        server: server.ok_or("--server is required")?,
        tolerance,
        separate,
    })
}
