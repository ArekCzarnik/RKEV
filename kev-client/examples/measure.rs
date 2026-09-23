//! Time a real checkpoint: what f16 buys, and what the prefix cache buys.
//!
//! ```bash
//! cargo run --release --features candle --example measure -- \
//!     --base ~/models/qwen-qwen3-0.6b-base --checkpoint ~/models/kev-0.6b
//! ```
//!
//! Every row is a median over `--repeat` passes, and each cold pass gets its own
//! state so it is a genuine cache miss rather than the same state twice. Two
//! controls are in the table on purpose, because a measurement here is easy to
//! fake by accident:
//!
//! * **cache off** — the same configuration with nothing kept between requests.
//!   If the cached row is not far below it, the cache is not doing what the
//!   number claims.
//! * **f16 against f32 on the answers** — a speed-up is only worth having if the
//!   probabilities survive it, so the largest difference is measured in the same
//!   run rather than assumed from elsewhere.
//!
//! The prefix threshold is forced to zero here. An attention-only base otherwise
//! skips the prefix below 384 state tokens, which would make the prefix rows
//! measure the packed path twice and agree beautifully.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use candle_core::{DType, Device};
use kev_client::{
    pointer_head, Answer, Backend, Choice, LocalEngine, Noul, Score, SystemOneRequest,
    SystemOneResponse,
};

fn main() -> ExitCode {
    let options = match parse() {
        Ok(options) => options,
        Err(message) => {
            eprintln!("{message}\n\n{USAGE}");
            return ExitCode::FAILURE;
        }
    };
    match run(&options) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

struct Options {
    base: PathBuf,
    checkpoint: Option<PathBuf>,
    head: Option<PathBuf>,
    repeat: usize,
    words: usize,
    batch: usize,
}

const USAGE: &str = "\
usage: measure --base <dir> [--checkpoint <dir>] [options]

  --head <file>     the pointer head, if not <checkpoint>/head.pt
  --repeat <n>      passes per row, median reported (default 3)
  --words <n>       length of the generated state, in words (default 400)
  --batch <n>       requests in the batched row (default 4)";

fn run(options: &Options) -> Result<(), Box<dyn std::error::Error>> {
    let checkpoint = options.checkpoint.as_deref();
    let head_path = options.head.clone().unwrap_or_else(|| {
        let dir = checkpoint.unwrap_or(&options.base);
        [dir.join("head.pt"), dir.join("head.safetensors")]
            .into_iter()
            .find(|path| path.exists())
            .unwrap_or_else(|| dir.join("head.pt"))
    });

    // One engine per configuration, because the knobs sit on the backend and a
    // loaded backend cannot be reconfigured behind the engine's mutex.
    let engine = |dtype: DType, prefix: bool, cache: usize| -> kev_client::Result<LocalEngine> {
        let backend = Backend::open_as(&options.base, checkpoint, Device::Cpu, dtype)?
            .with_prefix(prefix)
            .with_prefix_min_tokens(0)
            .with_prefix_cache(cache);
        Ok(LocalEngine::new(backend, pointer_head(&head_path)?))
    };

    let probe = Backend::open(&options.base, checkpoint)?;
    let hybrid = probe.is_hybrid();
    let default_dtype = probe.dtype();
    drop(probe);

    // Distinct states, so each first pass is a real miss. Same length, so the
    // passes are comparable.
    let states: Vec<String> = (0..options.repeat.max(options.batch))
        .map(|n| state(options.words, n))
        .collect();
    let requests: Vec<SystemOneRequest> = states.iter().map(|state| request(state)).collect();

    let reference = engine(DType::F32, true, 4)?.system_one_blocking(&requests[0])?;
    println!(
        "{}, {} questions over {} tokens of state, {} passes per row\n\
         the default precision on this device is {default_dtype:?}\n",
        if hybrid {
            "hybrid base (attention and Gated DeltaNet)"
        } else {
            "attention-only base"
        },
        reference.answers.len(),
        reference.usage.input_tokens,
        options.repeat,
    );

    let mut rows: Vec<(String, Vec<f64>)> = Vec::new();
    let mut measure = |label: &str, times: Vec<f64>| {
        println!("{:<44} {}", label, summarise(&times));
        rows.push((label.to_string(), times));
    };

    // --- what the prefix and its cache buy, at f32 ---
    let packed = engine(DType::F32, false, 0)?;
    measure(
        "f32, the state per question (packed)",
        times(&requests[..options.repeat], |r| {
            packed.system_one_blocking(r)
        })?,
    );
    let uncached = engine(DType::F32, true, 0)?;
    measure(
        "f32, the state once, nothing kept (control)",
        times(&requests[..options.repeat], |r| {
            uncached.system_one_blocking(r)
        })?,
    );
    let cached = engine(DType::F32, true, 8)?;
    // Warm every state first, then time the repeat: this row is meant to be all
    // hits, and the misses are the row above.
    for request in &requests[..options.repeat] {
        cached.system_one_blocking(request)?;
    }
    measure(
        "f32, the state found in the cache",
        times(&requests[..options.repeat], |r| {
            cached.system_one_blocking(r)
        })?,
    );

    // --- what f16 buys, if the answers survive it ---
    match engine(DType::F16, true, 0) {
        Ok(half) => {
            measure(
                "f16, the state once, nothing kept",
                times(&requests[..options.repeat], |r| half.system_one_blocking(r))?,
            );
            let theirs = half.system_one_blocking(&requests[0])?;
            println!(
                "{:<44} {:.5}",
                "f16 against f32, largest difference",
                difference(&reference, &theirs)
            );
        }
        Err(error) => println!("{:<44} {error}", "f16 is unavailable here"),
    }

    // --- the recurrence, where there is one ---
    if hybrid {
        let backend = Backend::open_as(&options.base, checkpoint, Device::Cpu, DType::F32)?
            .with_prefix(true)
            .with_prefix_min_tokens(0)
            .with_prefix_cache(0)
            .with_chunked_recurrence(false);
        let sequential = LocalEngine::new(backend, pointer_head(&head_path)?);
        measure(
            "f32, the delta rule token by token",
            times(&requests[..options.repeat], |r| {
                sequential.system_one_blocking(r)
            })?,
        );
    }

    // --- several requests at once ---
    if options.batch > 1 {
        let singly = engine(DType::F32, true, 0)?;
        let batched = engine(DType::F32, true, 0)?;
        let group = &requests[..options.batch];
        let one_at_a_time = Instant::now();
        for request in group {
            singly.system_one_blocking(request)?;
        }
        let one_at_a_time = one_at_a_time.elapsed().as_secs_f64() * 1000.0;
        let together = Instant::now();
        batched.system_one_batch_blocking(group)?;
        let together = together.elapsed().as_secs_f64() * 1000.0;
        println!(
            "\n{} requests one at a time {one_at_a_time:.0} ms, prefilled together \
             {together:.0} ms  ({:.2}x)",
            options.batch,
            one_at_a_time / together.max(f64::MIN_POSITIVE),
        );
    }

    // --- the ratios, which are the only part worth quoting ---
    let find = |needle: &str| {
        rows.iter()
            .find(|(label, _)| label.contains(needle))
            .map(|(_, times)| median(times))
    };
    println!();
    if let (Some(control), Some(hit)) = (find("nothing kept (control)"), find("found in the cache"))
    {
        println!("the cache saves {:.2}x on a repeated state", control / hit);
    }
    if let (Some(packed), Some(once)) = (
        find("per question (packed)"),
        find("nothing kept (control)"),
    ) {
        println!(
            "running the state once rather than per question: {:.2}x",
            packed / once
        );
    }
    if let (Some(exact), Some(half)) = (find("f32, the state once"), find("f16, the state once")) {
        println!("f16 rather than f32: {:.2}x", exact / half);
    }
    if let (Some(sequential), Some(chunked)) =
        (find("token by token"), find("nothing kept (control)"))
    {
        println!(
            "the delta rule in chunks rather than token by token: {:.2}x",
            sequential / chunked
        );
    }
    Ok(())
}

/// Time one pass per request, in milliseconds.
///
/// A failed pass is an error and not a fast one: swallowing it here would print a
/// very good number for doing nothing.
fn times<F>(requests: &[SystemOneRequest], mut answer: F) -> kev_client::Result<Vec<f64>>
where
    F: FnMut(&SystemOneRequest) -> kev_client::Result<SystemOneResponse>,
{
    requests
        .iter()
        .map(|request| {
            let started = Instant::now();
            answer(request)?;
            Ok(started.elapsed().as_secs_f64() * 1000.0)
        })
        .collect()
}

fn median(times: &[f64]) -> f64 {
    let mut sorted = times.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted.get(sorted.len() / 2).copied().unwrap_or(f64::NAN)
}

fn summarise(times: &[f64]) -> String {
    let min = times.iter().copied().fold(f64::INFINITY, f64::min);
    let max = times.iter().copied().fold(0.0, f64::max);
    format!(
        "{:>8.0} ms   (min {min:.0}, max {max:.0}, n={})",
        median(times),
        times.len()
    )
}

/// The largest difference between two answer sets to the same request.
fn difference(mine: &SystemOneResponse, theirs: &SystemOneResponse) -> f64 {
    mine.answers
        .iter()
        .filter_map(|(id, mine)| {
            let theirs = theirs.answer(id)?;
            Some(match (mine, theirs) {
                (Answer::Noul { noul: a }, Answer::Noul { noul: b }) => (a - b).abs(),
                _ => match (mine.probabilities(), theirs.probabilities()) {
                    (Some(a), Some(b)) => a
                        .iter()
                        .map(|(name, value)| {
                            (value - b.get(name).copied().unwrap_or(f64::NAN)).abs()
                        })
                        .fold(0.0, f64::max),
                    _ => f64::INFINITY,
                },
            })
        })
        .fold(0.0, f64::max)
}

/// A state of about `words` words, unique per `seed` so it misses the cache.
fn state(words: usize, seed: usize) -> String {
    const SENTENCES: [&str; 6] = [
        "The parcel was due on the third and has not arrived.",
        "Tracking last updated a week ago in the wrong city.",
        "I paid for next-day delivery and was charged twice.",
        "Support told me to wait and then closed the ticket.",
        "The shoes that did arrive are two sizes too small.",
        "I would like the refund processed today, please.",
    ];
    let mut text = format!("Ticket {seed}.");
    let mut index = seed;
    while text.split_whitespace().count() < words {
        text.push(' ');
        text.push_str(SENTENCES[index % SENTENCES.len()]);
        index += 1;
    }
    text
}

fn request(state: &str) -> SystemOneRequest {
    SystemOneRequest::new(state)
        .ask(
            "department",
            Choice::new("Which team should handle this?")
                .option("returns", "Exchanges, refunds, wrong or damaged items")
                .option("shipping", "Delivery status, delays, lost packages")
                .option("billing", "Charges, invoices, payment problems"),
        )
        .ask(
            "escalate",
            Noul::new("Does this need urgent human attention?"),
        )
        .ask(
            "refund",
            Noul::new("Is the customer asking for money back?"),
        )
        .ask(
            "frustration",
            Score::new("How frustrated is the customer?")
                .level("Calm")
                .level("Frustrated")
                .level("Very angry"),
        )
        .ask(
            "urgency",
            Score::new("How urgent is this ticket?")
                .level("can wait")
                .level("this week")
                .level("today"),
        )
}

fn parse() -> Result<Options, String> {
    let mut base = None;
    let mut checkpoint = None;
    let mut head = None;
    let mut repeat = 3usize;
    let mut words = 400usize;
    let mut batch = 4usize;

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
            "--repeat" => repeat = value()?.parse().map_err(|e| format!("--repeat: {e}"))?,
            "--words" => words = value()?.parse().map_err(|e| format!("--words: {e}"))?,
            "--batch" => batch = value()?.parse().map_err(|e| format!("--batch: {e}"))?,
            "-h" | "--help" => return Err(String::from("measure")),
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if repeat == 0 {
        return Err(String::from("--repeat has to be at least 1"));
    }
    Ok(Options {
        base: base.ok_or("--base is required")?,
        checkpoint,
        head,
        repeat,
        words,
        batch,
    })
}
