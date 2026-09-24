//! Check a real checkpoint against nothing but itself and common sense.
//!
//! There are no reference numbers here, so this asks the two questions that can
//! be answered without them:
//!
//! * **Does it hang together?** The same request answered along different paths
//!   has to give the same numbers: all questions at once against one at a time,
//!   the state prefilled once against per question, and — on a recurrent base —
//!   the delta rule in chunks against token by token. These compare the
//!   implementation with itself, but they walk different code, and a mistake in
//!   the layout, the mask or the recurrence shows up as a disagreement.
//! * **Does it mean anything?** A handful of tickets whose answer is not in
//!   doubt. A checkpoint that scores 0.87 on held-out data should get nearly all
//!   of them; an implementation that loads the weights but gets the prompt, the
//!   rotary embedding or the readout wrong will sit near chance with flat
//!   distributions, which is what this makes visible.
//!
//! Neither is parity. Parity needs the server, and `examples/parity.rs` is that.
//! This is what tells you whether it is worth setting one up.
//!
//! ```bash
//! cargo run --features candle --example sanity -- \
//!     --base ~/kev/qwen3-0.6b-base --checkpoint ~/kev/kev-0.6b
//! ```

use std::path::PathBuf;
use std::process::ExitCode;

use candle_core::Device;
use rkev::{
    device, pointer_head, Answer, Backend, Choice, LocalEngine, Noul, Quantisation, Score,
    SystemOneRequest,
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
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
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
    tolerance: f64,
    /// Where the backbone runs; the precision follows it.
    device: Device,
    /// Quantise the projections, to see what that costs the answers.
    quantise: Option<Quantisation>,
}

/// One case whose answer is not in doubt.
struct Case {
    ticket: &'static str,
    question: fn() -> SystemOneRequest,
    /// What a trained checkpoint has to say about it.
    expected: Expected,
}

enum Expected {
    /// The most likely option, by name.
    Choice(&'static str),
    /// Yes or no, read off `p(yes)`.
    Noul(bool),
    /// Which end of the scale the mean level has to sit on.
    Score {
        above: Option<f64>,
        below: Option<f64>,
    },
}

fn cases() -> Vec<Case> {
    fn department(state: &str) -> SystemOneRequest {
        SystemOneRequest::new(state.to_string()).ask(
            "department",
            Choice::new("Which team should handle this?")
                .option("returns", "Exchanges, refunds, wrong or damaged items")
                .option("shipping", "Delivery status, delays, lost packages")
                .option("billing", "Charges, invoices, payment problems"),
        )
    }
    vec![
        Case {
            ticket: "My parcel never arrived and the tracking has not moved in a week.",
            question: || {
                department("My parcel never arrived and the tracking has not moved in a week.")
            },
            expected: Expected::Choice("shipping"),
        },
        Case {
            ticket: "I was charged twice for order 4411. Please refund one of them.",
            question: || {
                department("I was charged twice for order 4411. Please refund one of them.")
            },
            expected: Expected::Choice("billing"),
        },
        Case {
            ticket: "The shirt is two sizes too small. Can I exchange it?",
            question: || department("The shirt is two sizes too small. Can I exchange it?"),
            expected: Expected::Choice("returns"),
        },
        Case {
            ticket: "THIRD time writing. Nobody answers. I want my money back today.",
            question: || {
                SystemOneRequest::new(
                    "THIRD time writing. Nobody answers. I want my money back today.",
                )
                .ask(
                    "escalate",
                    Noul::new("Does this need urgent human attention?"),
                )
            },
            expected: Expected::Noul(true),
        },
        Case {
            ticket: "Just wanted to say thank you, the shoes are lovely.",
            question: || {
                SystemOneRequest::new("Just wanted to say thank you, the shoes are lovely.").ask(
                    "escalate",
                    Noul::new("Does this need urgent human attention?"),
                )
            },
            expected: Expected::Noul(false),
        },
        Case {
            ticket: "I am absolutely furious about this.",
            question: || {
                SystemOneRequest::new("I am absolutely furious about this.").ask(
                    "frustration",
                    Score::new("How frustrated is the customer?")
                        .level("Calm")
                        .level("Frustrated")
                        .level("Very angry"),
                )
            },
            expected: Expected::Score {
                above: Some(1.4),
                below: None,
            },
        },
        Case {
            ticket: "No rush at all, whenever you get round to it.",
            question: || {
                SystemOneRequest::new("No rush at all, whenever you get round to it.").ask(
                    "urgency",
                    Score::new("How urgent is this ticket?")
                        .level("can wait")
                        .level("this week")
                        .level("today"),
                )
            },
            expected: Expected::Score {
                above: None,
                below: Some(0.6),
            },
        },
    ]
}

fn run(options: &Options) -> Result<bool, Box<dyn std::error::Error>> {
    let checkpoint = options.checkpoint.as_deref();
    let head = options.head.clone().unwrap_or_else(|| {
        let dir = checkpoint.unwrap_or(&options.base);
        // kev ships head.pt; an exported head may sit beside it as safetensors.
        [dir.join("head.pt"), dir.join("head.safetensors")]
            .into_iter()
            .find(|path| path.exists())
            .unwrap_or_else(|| dir.join("head.pt"))
    });
    let open = || -> Result<Backend, rkev::Error> {
        match options.quantise {
            None => Backend::open_on(&options.base, checkpoint, options.device.clone()),
            Some(quantise) => Backend::open_with(
                &options.base,
                checkpoint,
                options.device.clone(),
                candle_core::DType::F32,
                Some(quantise),
            ),
        }
    };
    let engine = || -> Result<LocalEngine, rkev::Error> {
        Ok(LocalEngine::new(open()?, pointer_head(&head)?))
    };

    let backend = open()?;
    println!(
        "{:?}\n{} the layers are {}, the head expects {} hidden units\n",
        backend,
        if backend.is_hybrid() {
            "hybrid base:"
        } else {
            "attention-only base:"
        },
        if backend.is_hybrid() {
            "attention mixed with Gated DeltaNet, so one row per question"
        } else {
            "all attention, so one masked pass per request"
        },
        backend.hidden_size(),
    );
    if let Some(isolation) = rkev::option_isolation(&head).ok().flatten() {
        println!("head.pt says option_isolation = {isolation}");
        if isolation {
            println!("  (pass --option-isolation to match it; without it the prompt differs)\n");
        }
    }

    // --- does it mean anything ---
    println!("{:<62} {:>31}  expected", "ticket", "answer");
    let mut hits = 0;
    let cases = cases();
    for case in &cases {
        let response = engine()?.system_one_blocking(&(case.question)())?;
        let (id, answer) = response
            .answers
            .iter()
            .next()
            .ok_or("the engine answered nothing")?;
        let (shown, ok) = judge(answer, &case.expected);
        hits += usize::from(ok);
        println!(
            "{:<62} {:>31}  {} {}",
            shorten(case.ticket, 60),
            shown,
            expected_of(&case.expected, id),
            if ok { "ok" } else { "MISSED" }
        );
    }
    println!("\n{hits} of {} obvious cases", cases.len());

    // --- is the pointer head the right way round ---
    //
    // Which projection reads `<decide>` and which reads each `</opt>` comes from
    // head.pt's own naming, and swapping them gives a different distribution that
    // is just as plausible: no shape and no self-consistency check can separate
    // them, because both sides would be swapped alike. A *trained* head can. Only
    // worth the passes if the ordinary orientation answered these cases at all.
    if hits + 1 >= cases.len() {
        let swapped = || -> Result<LocalEngine, rkev::Error> {
            Ok(LocalEngine::new(open()?, pointer_head(&head)?.swapped()))
        };
        let mut wrong_way = 0;
        for case in &cases {
            let response = swapped()?.system_one_blocking(&(case.question)())?;
            let answer = response
                .answers
                .values()
                .next()
                .ok_or("the engine answered nothing")?;
            wrong_way += usize::from(judge(answer, &case.expected).1);
        }
        println!(
            "{wrong_way} of {} with the head's two projections swapped",
            cases.len()
        );
        if wrong_way < hits {
            println!(
                "  so the projections are the right way round: exchanging them costs \
                 {} case(s).",
                hits - wrong_way
            );
        } else {
            println!(
                "  WARNING: swapping them costs nothing here, so these cases do not \
                 pin the orientation. Either the head is untrained, or the two \
                 projections are near enough alike that only a recorded server \
                 response can settle it."
            );
        }
    }

    // --- does it hang together ---
    println!();
    let mut consistent = true;
    let request = SystemOneRequest::new(
        "Shoes arrived two weeks late and in the wrong size. \
         Also I see two charges on my card.",
    )
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
        "frustration",
        Score::new("How frustrated is the customer?")
            .level("Calm")
            .level("Frustrated")
            .level("Very angry"),
    );

    let reference = engine()?.system_one_blocking(&request)?;
    println!("the worked example from the Kev README:");
    for (id, answer) in &reference.answers {
        println!("  {id:<14} {}", describe(answer));
    }
    println!("  (the README publishes 0.47/0.28/0.25, p(yes) 0.93, score 1.44 for Kev-4B in bf16)");

    println!();
    let mut compare = |what: &str, other: &rkev::SystemOneResponse| {
        let worst = reference
            .answers
            .iter()
            .filter_map(|(id, mine)| {
                let theirs = other.answer(id)?;
                Some(distance(mine, theirs))
            })
            .fold(0.0f64, f64::max);
        let ok = worst <= options.tolerance;
        consistent &= ok;
        println!(
            "{what:<52} largest difference {worst:.5}  {}",
            if ok { "ok" } else { "DISAGREES" }
        );
    };

    compare(
        "one question at a time against all at once",
        &engine()?.system_one_separate_blocking(&request)?,
    );
    let packed = LocalEngine::new(open()?.with_prefix(false), pointer_head(&head)?)
        .system_one_blocking(&request)?;
    compare("the state per question against prefilled once", &packed);
    if backend.is_hybrid() {
        let sequential =
            LocalEngine::new(open()?.with_chunked_recurrence(false), pointer_head(&head)?)
                .system_one_blocking(&request)?;
        compare(
            "the delta rule token by token against in chunks",
            &sequential,
        );
    }

    println!(
        "\n{}",
        if consistent {
            "the paths agree. What this cannot tell you is whether they agree with the \
             server - for that, examples/parity.rs."
        } else {
            "the paths disagree. In f32 on a CPU they are exact and a difference is a \
             bug in the layout, the mask or the recurrence. Quantised, or on a GPU, the \
             kernels reorder the arithmetic and a few thousandths are float \
             associativity rather than a fault - run it again dense on the CPU to tell \
             the two apart."
        }
    );
    Ok(consistent && hits + 1 >= cases.len())
}

fn judge(answer: &Answer, expected: &Expected) -> (String, bool) {
    match (answer, expected) {
        (Answer::Choice { choice, .. }, Expected::Choice(wanted)) => {
            (describe(answer), choice == wanted)
        }
        (Answer::Noul { noul }, Expected::Noul(wanted)) => {
            (describe(answer), (*noul > 0.5) == *wanted)
        }
        (Answer::Score { score, .. }, Expected::Score { above, below }) => (
            describe(answer),
            above.map_or(true, |bound| *score > bound)
                && below.map_or(true, |bound| *score < bound),
        ),
        _ => (describe(answer), false),
    }
}

fn describe(answer: &Answer) -> String {
    match answer {
        Answer::Noul { noul } => format!("p(yes) {noul:.2}"),
        Answer::Choice {
            choice,
            confidence,
            probabilities,
        } => format!(
            "{choice} ({:.2}, confidence {confidence:.2})",
            probabilities.get(choice).copied().unwrap_or(f64::NAN)
        ),
        Answer::Score {
            score, confidence, ..
        } => format!("level {score:.2} (confidence {confidence:.2})"),
    }
}

fn expected_of(expected: &Expected, id: &str) -> String {
    match expected {
        Expected::Choice(name) => format!("{id} = {name}"),
        Expected::Noul(yes) => format!("{id} = {}", if *yes { "yes" } else { "no" }),
        Expected::Score { above, below } => match (above, below) {
            (Some(bound), _) => format!("{id} > {bound}"),
            (_, Some(bound)) => format!("{id} < {bound}"),
            _ => id.to_string(),
        },
    }
}

/// The largest difference between two answers to the same question.
fn distance(mine: &Answer, theirs: &Answer) -> f64 {
    match (mine.probabilities(), theirs.probabilities()) {
        (Some(mine), Some(theirs)) => mine
            .iter()
            .map(|(name, value)| (value - theirs.get(name).copied().unwrap_or(f64::NAN)).abs())
            .fold(0.0, f64::max),
        _ => match (mine.as_noul(), theirs.as_noul()) {
            (Some(mine), Some(theirs)) => (mine - theirs).abs(),
            _ => f64::INFINITY,
        },
    }
}

fn shorten(text: &str, width: usize) -> String {
    if text.len() <= width {
        return text.to_string();
    }
    format!("{}…", &text[..width - 1])
}

const USAGE: &str = "\
usage: sanity --base <dir> [--checkpoint <dir>] [--head <file>] [--tolerance <f64>]

  --base        the base model directory (config.json and its safetensors)
  --checkpoint  the Kev checkpoint: adapter, head.pt, tokenizer
  --head        the pointer head, if it is not <checkpoint>/head.pt
  --tolerance   largest difference to accept between paths (default 0.001)
  --device      cpu|metal; metal needs --features metal (macOS only)
  --quantise    q4k|q5k|q6k|q8_0, to see what quantising costs the answers";

fn parse() -> Result<Options, String> {
    let mut base = None;
    let mut checkpoint = None;
    let mut head = None;
    let mut tolerance = 0.001;
    let mut chosen = Device::Cpu;
    let mut quantise = None;

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
            "--tolerance" => {
                tolerance = value()?.parse().map_err(|e| format!("--tolerance: {e}"))?
            }
            "--device" => chosen = device(&value()?).map_err(|e| e.to_string())?,
            "--quantise" | "--quantize" => {
                quantise = Some(Quantisation::from_name(&value()?).map_err(|e| e.to_string())?)
            }
            "-h" | "--help" => return Err(String::from("sanity")),
            other => return Err(format!("unknown argument {other}")),
        }
    }
    Ok(Options {
        base: base.ok_or("--base is required")?,
        checkpoint,
        head,
        tolerance,
        device: chosen,
        quantise,
    })
}
