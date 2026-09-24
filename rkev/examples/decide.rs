//! Answer System One requests with a local checkpoint.
//!
//! This is the Kev server's job done in process. It loads a base model plus a
//! Kev checkpoint, answers the requests you hand it, and prints the answers —
//! either for reading, or as the JSON the server would have replied with.
//!
//! ```bash
//! # the questions and the state in one request file, as you would POST it
//! cargo run --release --features candle --example decide -- \
//!     --base ~/models/qwen3-0.6b-base --checkpoint ~/models/kev-0.6b \
//!     --request request.json
//!
//! # or just a state, against the README's triage questions
//! cargo run --release --features candle --example decide -- \
//!     --base <base> --checkpoint <kev> --state "My parcel never arrived."
//!
//! # a state per line on stdin, one JSON answer per line out: the model loads once
//! cat tickets.txt | cargo run --release --features candle --example decide -- \
//!     --base <base> --checkpoint <kev> --questions questions.json --lines --json
//! ```
//!
//! Diagnostics go to stderr, answers to stdout, so `--json` output can be piped
//! straight into `jq`.

use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use candle_core::{DType, Device};
use rkev::{
    answers_json, device, option_isolation, pointer_head, Answer, Backend, Choice, IndexMap,
    LocalEngine, Noul, Quantisation, Question, Score, SystemOneRequest, SystemOneResponse,
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

fn run(options: &Options) -> Result<(), Box<dyn std::error::Error>> {
    let loading = Instant::now();
    let head_path = options.head.clone().unwrap_or_else(|| {
        let dir = options.checkpoint.as_ref().unwrap_or(&options.base);
        // kev ships head.pt; an exported head may sit beside it as safetensors.
        [dir.join("head.pt"), dir.join("head.safetensors")]
            .into_iter()
            .find(|path| path.exists())
            .unwrap_or_else(|| dir.join("head.pt"))
    });
    let checkpoint = options.checkpoint.as_deref();

    let mut backend = match (options.dtype, options.quantise) {
        (dtype, Some(quantise)) => Backend::open_with(
            &options.base,
            checkpoint,
            options.device.clone(),
            dtype.unwrap_or(DType::F32),
            Some(quantise),
        )?,
        (Some(dtype), None) => {
            Backend::open_as(&options.base, checkpoint, options.device.clone(), dtype)?
        }
        (None, None) => Backend::open_on(&options.base, checkpoint, options.device.clone())?,
    };
    // Every state a --lines run has seen stays available to the next one.
    if options.lines {
        backend = backend.with_prefix_cache(16);
    }
    let hybrid = backend.is_hybrid();
    let dtype = backend.dtype();

    // A checkpoint records the layout it was trained on, and serving the other
    // one is a silently different prompt rather than an error.
    let isolation = match options.isolation {
        Some(chosen) => Some(chosen),
        None => option_isolation(&head_path)?,
    };
    let mut engine = LocalEngine::new(backend, pointer_head(&head_path)?);
    if let Some(isolation) = isolation {
        engine = engine.with_option_isolation(isolation);
    }
    if let Some(model) = &options.model {
        engine = engine.with_model(model.clone());
    }
    eprintln!(
        "{} in {:?}{}, loaded in {:.1}s{}",
        if hybrid {
            "hybrid base (attention and Gated DeltaNet)"
        } else {
            "attention-only base"
        },
        dtype,
        match isolation {
            Some(true) => ", options isolated",
            Some(false) => "",
            None => ", option layout unknown",
        },
        loading.elapsed().as_secs_f64(),
        if options.lines { ", ready" } else { "" }
    );

    let questions = match &options.questions {
        Some(path) => serde_json::from_str::<IndexMap<String, Question>>(&read(path)?)?,
        None => triage_questions(),
    };

    // --- one state per line, model already loaded ---
    if options.lines {
        let input = std::io::stdin();
        let mut output = std::io::stdout().lock();
        for line in input.lock().lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let request = request_with(&line, &questions);
            let answered = answer(&engine, options, &request)?;
            write(&mut output, options, &answered)?;
            output.flush()?;
        }
        return Ok(());
    }

    // --- whatever was asked for once ---
    let mut requests = Vec::new();
    for path in &options.requests {
        requests.push(serde_json::from_str::<SystemOneRequest>(&read(path)?)?);
    }
    if let Some(state) = &options.state {
        requests.push(request_with(state, &questions));
    }
    if requests.is_empty() {
        return Err("nothing to answer: pass --request, --state, or --lines".into());
    }

    let mut output = std::io::stdout().lock();
    if let Some(question) = &options.permute {
        // The permutation endpoint asks one question under several option orders.
        for request in &requests {
            let report =
                engine.permute_blocking(request, question, options.rounds, options.seed)?;
            writeln!(output, "{report:#}")?;
        }
        return Ok(());
    }
    if requests.len() > 1 && !options.separate {
        // Several states, one prefill pass over all of them.
        for answered in engine.system_one_batch_blocking(&requests)? {
            write(&mut output, options, &answered)?;
        }
        return Ok(());
    }
    for request in &requests {
        let answered = answer(&engine, options, request)?;
        write(&mut output, options, &answered)?;
    }
    Ok(())
}

fn answer(
    engine: &LocalEngine,
    options: &Options,
    request: &SystemOneRequest,
) -> rkev::Result<SystemOneResponse> {
    if options.separate {
        engine.system_one_separate_blocking(request)
    } else {
        engine.system_one_blocking(request)
    }
}

fn request_with(state: &str, questions: &IndexMap<String, Question>) -> SystemOneRequest {
    // A state that parses as JSON is sent as JSON: Kev renders objects and
    // arrays into labelled text, and quoting one would change the prompt.
    let mut request = match serde_json::from_str::<serde_json::Value>(state) {
        Ok(value) if value.is_object() || value.is_array() => SystemOneRequest::new(value),
        _ => SystemOneRequest::new(state),
    };
    request.questions = questions.clone();
    request
}

/// The worked example from the Kev README, for when no questions are given.
fn triage_questions() -> IndexMap<String, Question> {
    let mut questions = IndexMap::new();
    questions.insert(
        String::from("department"),
        Question::Choice(
            Choice::new("Which team should handle this?")
                .option("returns", "Exchanges, refunds, wrong or damaged items")
                .option("shipping", "Delivery status, delays, lost packages")
                .option("billing", "Charges, invoices, payment problems"),
        ),
    );
    questions.insert(
        String::from("escalate"),
        Question::Noul(Noul::new("Does this need urgent human attention?")),
    );
    questions.insert(
        String::from("frustration"),
        Question::Score(
            Score::new("How frustrated is the customer?")
                .level("Calm")
                .level("Frustrated")
                .level("Very angry"),
        ),
    );
    questions
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

fn write(
    out: &mut impl Write,
    options: &Options,
    response: &SystemOneResponse,
) -> std::io::Result<()> {
    if options.json {
        writeln!(out, "{}", response_json(response))
    } else {
        for (id, answer) in &response.answers {
            writeln!(out, "{}", describe(id, answer))?;
        }
        if let Some(latency) = response.latency_ms {
            writeln!(
                out,
                "{:<14} {} tokens in, {:.0} ms\n",
                "", response.usage.input_tokens, latency
            )?;
        }
        Ok(())
    }
}

/// The response as the server sends it. The answers are serialised the way the
/// Python serialises them — that is what `usage.output_tokens` counts, so the
/// tests pin it — and the envelope is assembled around them here.
fn response_json(response: &SystemOneResponse) -> String {
    format!(
        "{{\"model\": {}, \"answers\": {}, \"usage\": {{\"input_tokens\": {}, \
         \"output_tokens\": {}}}, \"latency_ms\": {}}}",
        serde_json::to_string(&response.model).unwrap_or_else(|_| String::from("null")),
        answers_json(&response.answers),
        response.usage.input_tokens,
        response.usage.output_tokens,
        response
            .latency_ms
            .map_or_else(|| String::from("null"), |ms| format!("{ms:.2}")),
    )
}

fn describe(id: &str, answer: &Answer) -> String {
    match answer {
        Answer::Noul { noul } => format!(
            "{id:<14} {:<12} {noul:.2}",
            if *noul > 0.5 { "yes" } else { "no" }
        ),
        Answer::Choice {
            choice,
            confidence,
            probabilities,
        } => {
            let mut lines = vec![format!(
                "{id:<14} {choice:<12} {:.2}   confidence {confidence:.2}",
                probabilities.get(choice).copied().unwrap_or(f64::NAN)
            )];
            // The rest of the distribution, heaviest first: the point of Kev is
            // that it is there.
            let mut rest: Vec<_> = probabilities.iter().filter(|(k, _)| *k != choice).collect();
            rest.sort_by(|a, b| b.1.total_cmp(a.1));
            for (name, probability) in rest {
                lines.push(format!("{:<14} {name:<12} {probability:.2}", ""));
            }
            lines.join("\n")
        }
        Answer::Score {
            score,
            confidence,
            legend,
            probabilities,
        } => {
            let nearest = legend
                .get(&format!("{}", score.round() as i64))
                .map(String::as_str)
                .unwrap_or("");
            // "level" because a score is a position on the scale, not a
            // probability like the numbers under it.
            let mut lines = vec![format!(
                "{id:<14} {nearest:<12} level {score:.2}   confidence {confidence:.2}"
            )];
            for (index, probability) in probabilities {
                let level = legend.get(index).map(String::as_str).unwrap_or(index);
                lines.push(format!("{:<14} {level:<12} {probability:.2}", ""));
            }
            lines.join("\n")
        }
    }
}

// ---------------------------------------------------------------------------
// Arguments
// ---------------------------------------------------------------------------

struct Options {
    base: PathBuf,
    checkpoint: Option<PathBuf>,
    head: Option<PathBuf>,
    dtype: Option<DType>,
    /// Where the backbone runs. The precision follows it unless --dtype says.
    device: Device,
    /// Quantise the projections after merging the adapter, for the bandwidth.
    quantise: Option<Quantisation>,
    model: Option<String>,
    requests: Vec<PathBuf>,
    state: Option<String>,
    questions: Option<PathBuf>,
    lines: bool,
    separate: bool,
    json: bool,
    isolation: Option<bool>,
    permute: Option<String>,
    rounds: u8,
    seed: u64,
}

const USAGE: &str = "\
usage: decide --base <dir> [--checkpoint <dir>] <what to answer> [options]

the model
  --base <dir>           the base model: config.json and its safetensors
  --checkpoint <dir>     the Kev checkpoint: LoRA adapter, head.pt, tokenizer
  --head <file>          the pointer head, if not <checkpoint>/head.pt
  --dtype f32|f16|bf16   default: f32 on a CPU, bf16 on a GPU, as kev.serve picks it
  --device cpu|metal     where to run; metal needs --features metal (macOS)
  --quantise q4k|q5k|q6k|q8_0
                         quantise the projections after the merge; CPU and f32
  --model <name>         the name to report back in the answers

what to answer
  --request <path|->     a System One request as JSON; repeat for a batch
  --state <text|@file|-> a state to ask --questions about
  --questions <path>     the questions map as JSON; without it, the README's
                         triage questions (department, escalate, frustration)
  --lines                a state per line on stdin, answered as they arrive

how
  --separate             one pass per question, as /v1/systemone/separate does
  --permute <question>   run one choice question under several option orders
  --rounds <n>           orders to try, 1..=64 (default 8)
  --seed <n>             the order seed (default 0)
  --option-isolation, --no-option-isolation
                         override the layout head.pt records
  --json                 print what the server would have replied";

fn parse() -> Result<Options, String> {
    let mut options = Options {
        base: PathBuf::new(),
        checkpoint: None,
        head: None,
        dtype: None,
        device: Device::Cpu,
        quantise: None,
        model: None,
        requests: Vec::new(),
        state: None,
        questions: None,
        lines: false,
        separate: false,
        json: false,
        isolation: None,
        permute: None,
        rounds: 8,
        seed: 0,
    };
    let mut base = None;

    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        let mut value = || {
            arguments
                .next()
                .ok_or_else(|| format!("{argument} needs a value"))
        };
        match argument.as_str() {
            "--base" => base = Some(PathBuf::from(value()?)),
            "--checkpoint" => options.checkpoint = Some(PathBuf::from(value()?)),
            "--head" => options.head = Some(PathBuf::from(value()?)),
            "--dtype" => {
                options.dtype = Some(match value()?.as_str() {
                    "f32" | "fp32" | "float32" => DType::F32,
                    "f16" | "fp16" | "float16" => DType::F16,
                    "bf16" | "bfloat16" => DType::BF16,
                    other => return Err(format!("--dtype {other}: f32, f16 or bf16")),
                })
            }
            "--device" => options.device = device(&value()?).map_err(|e| e.to_string())?,
            "--quantise" | "--quantize" => {
                options.quantise =
                    Some(Quantisation::from_name(&value()?).map_err(|e| e.to_string())?)
            }
            "--model" => options.model = Some(value()?),
            "--request" => options.requests.push(PathBuf::from(value()?)),
            "--state" => options.state = Some(value()?),
            "--questions" => options.questions = Some(PathBuf::from(value()?)),
            "--lines" => options.lines = true,
            "--separate" => options.separate = true,
            "--json" => options.json = true,
            "--option-isolation" => options.isolation = Some(true),
            "--no-option-isolation" => options.isolation = Some(false),
            "--permute" => options.permute = Some(value()?),
            "--rounds" => {
                options.rounds = value()?.parse().map_err(|e| format!("--rounds: {e}"))?
            }
            "--seed" => options.seed = value()?.parse().map_err(|e| format!("--seed: {e}"))?,
            "-h" | "--help" => return Err(String::from("decide")),
            other => return Err(format!("unknown argument {other}")),
        }
    }
    options.base = base.ok_or("--base is required")?;
    if let Some(state) = options.state.take() {
        options.state = Some(read_state(&state).map_err(|e| e.to_string())?);
    }
    Ok(options)
}

/// A state given as `@file`, `-` for stdin, or the text itself.
fn read_state(given: &str) -> std::io::Result<String> {
    match given.strip_prefix('@') {
        Some(path) => Ok(std::fs::read_to_string(path)?.trim_end().to_string()),
        None if given == "-" => read(Path::new("-")).map(|text| text.trim_end().to_string()),
        None => Ok(given.to_string()),
    }
}

fn read(path: &Path) -> std::io::Result<String> {
    if path == Path::new("-") {
        let mut text = String::new();
        std::io::Read::read_to_string(&mut std::io::stdin(), &mut text)?;
        return Ok(text);
    }
    std::fs::read_to_string(path)
}
