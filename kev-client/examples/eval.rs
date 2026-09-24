//! Measure a checkpoint on your own labelled tickets.
//!
//! Parity asks whether this engine agrees with the reference. This asks the other
//! question, the one that decides whether a checkpoint is any use to you: on your
//! tickets, in your language, with your questions, how often is it right — and is
//! it right in a way you can route on?
//!
//! ```bash
//! cargo run --release --example eval -- \
//!     --base ~/models/qwen-qwen3-0.6b-base --checkpoint ~/models/kev-0.6b \
//!     --questions tests/eval/questions.json --records tests/eval/tickets.jsonl
//! ```
//!
//! `tests/eval/README.md` has the record format. Per question it reports the
//! measure that fits the type, and then accuracy by confidence — which is what a
//! routing threshold comes from, and the reason to run a model that answers with
//! a distribution instead of a label.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::ExitCode;

use kev_client::{
    option_isolation, pointer_head, Answer, Backend, IndexMap, LocalEngine, Question,
    SystemOneRequest, SystemOneResponse,
};
use serde_json::Value;

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
    questions: Option<PathBuf>,
    request: Option<PathBuf>,
    records: PathBuf,
    batch: usize,
    limit: Option<usize>,
    errors: usize,
    json: bool,
}

/// One labelled record: the state to answer, and what you consider right.
struct Record {
    line: usize,
    state: Value,
    labels: IndexMap<String, Value>,
}

/// What one question's answers came to over the whole set.
#[derive(Default)]
struct Tally {
    kind: &'static str,
    /// Records this question was labelled on.
    scored: usize,
    hits: usize,
    /// `(confidence, hit)` per scored record, for the calibration table.
    confidence: Vec<(f64, bool)>,
    /// `choice` only: label -> answer -> count.
    confusion: BTreeMap<String, BTreeMap<String, usize>>,
    /// `noul` only: p(yes) split by the label.
    yes: Vec<f64>,
    no: Vec<f64>,
    /// `score` only: the absolute error per record.
    errors: Vec<f64>,
    /// The confident mistakes, worst first: (confidence, line, label, answer).
    misses: Vec<(f64, usize, String, String)>,
}

fn run(options: &Options) -> Result<(), Box<dyn std::error::Error>> {
    // --- the questions, and the records they are asked of ---
    let questions: IndexMap<String, Question> = match (&options.questions, &options.request) {
        (Some(path), _) => serde_json::from_str(&read(path)?)?,
        (None, Some(path)) => serde_json::from_str::<SystemOneRequest>(&read(path)?)?.questions,
        (None, None) => return Err("--questions or --request is required".into()),
    };
    let records = load(&options.records, &questions, options.limit)?;
    if records.is_empty() {
        return Err(format!("{} holds no records", options.records.display()).into());
    }

    // --- the engine ---
    let checkpoint = options.checkpoint.as_deref();
    let head = options.head.clone().unwrap_or_else(|| {
        let dir = checkpoint.unwrap_or(&options.base);
        [dir.join("head.pt"), dir.join("head.safetensors")]
            .into_iter()
            .find(|path| path.exists())
            .unwrap_or_else(|| dir.join("head.pt"))
    });
    let backend = Backend::open(&options.base, checkpoint)?;
    let hybrid = backend.is_hybrid();
    let dtype = backend.dtype();
    let mut engine = LocalEngine::new(backend, pointer_head(&head)?);
    if let Some(isolation) = option_isolation(&head)? {
        engine = engine.with_option_isolation(isolation);
    }
    eprintln!(
        "{} in {dtype:?}, {} records, {} questions",
        if hybrid {
            "hybrid base"
        } else {
            "attention-only base"
        },
        records.len(),
        questions.len(),
    );

    // --- answer everything ---
    let mut tallies: IndexMap<String, Tally> = IndexMap::new();
    let mut answered = 0;
    for chunk in records.chunks(options.batch.max(1)) {
        let requests: Vec<SystemOneRequest> = chunk
            .iter()
            .map(|record| {
                let mut request = SystemOneRequest::new(record.state.clone());
                request.questions = questions.clone();
                request
            })
            .collect();
        let responses = if requests.len() == 1 {
            vec![engine.system_one_blocking(&requests[0])?]
        } else {
            engine.system_one_batch_blocking(&requests)?
        };
        for (record, response) in chunk.iter().zip(&responses) {
            score(record, response, &questions, &mut tallies);
        }
        answered += chunk.len();
        if answered % 25 == 0 || answered == records.len() {
            eprintln!("  {answered} of {}", records.len());
        }
    }

    if options.json {
        println!("{}", as_json(&tallies, records.len()));
        return Ok(());
    }
    report(&tallies, records.len(), options.errors);
    Ok(())
}

/// Compare one response with one record's labels.
fn score(
    record: &Record,
    response: &SystemOneResponse,
    questions: &IndexMap<String, Question>,
    tallies: &mut IndexMap<String, Tally>,
) {
    for (id, label) in &record.labels {
        let Some(answer) = response.answer(id) else {
            continue;
        };
        let tally = tallies.entry(id.clone()).or_default();
        tally.kind = match questions.get(id) {
            Some(Question::Noul(_)) => "noul",
            Some(Question::Choice(_)) => "choice",
            Some(Question::Score(_)) => "score",
            None => "?",
        };
        tally.scored += 1;

        match answer {
            Answer::Choice {
                choice,
                confidence,
                probabilities,
            } => {
                let wanted = label.as_str().unwrap_or_default().to_string();
                let hit = *choice == wanted;
                tally.hits += usize::from(hit);
                tally.confidence.push((*confidence, hit));
                *tally
                    .confusion
                    .entry(wanted.clone())
                    .or_default()
                    .entry(choice.clone())
                    .or_default() += 1;
                if !hit {
                    // How sure it was about the wrong option, which is what makes
                    // a mistake worth looking at.
                    let sure = probabilities.get(choice).copied().unwrap_or(*confidence);
                    tally
                        .misses
                        .push((sure, record.line, wanted, choice.clone()));
                }
            }
            Answer::Noul { noul } => {
                let wanted = truthy(label);
                let hit = (*noul > 0.5) == wanted;
                tally.hits += usize::from(hit);
                // For a yes/no answer the probability is the confidence, read
                // towards whichever side it fell on.
                tally
                    .confidence
                    .push((if *noul > 0.5 { *noul } else { 1.0 - noul }, hit));
                if wanted {
                    tally.yes.push(*noul)
                } else {
                    tally.no.push(*noul)
                }
                if !hit {
                    tally.misses.push((
                        if *noul > 0.5 { *noul } else { 1.0 - noul },
                        record.line,
                        String::from(if wanted { "yes" } else { "no" }),
                        format!("{noul:.2}"),
                    ));
                }
            }
            Answer::Score {
                score,
                confidence,
                legend,
                ..
            } => {
                let wanted = level(label, legend);
                let hit = wanted.is_some_and(|wanted| (score.round() as i64) == wanted);
                tally.hits += usize::from(hit);
                tally.confidence.push((*confidence, hit));
                if let Some(wanted) = wanted {
                    tally.errors.push((score - wanted as f64).abs());
                }
                if !hit {
                    tally.misses.push((
                        *confidence,
                        record.line,
                        wanted.map_or_else(|| label.to_string(), |w| w.to_string()),
                        format!("{score:.2}"),
                    ));
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

fn report(tallies: &IndexMap<String, Tally>, records: usize, errors: usize) {
    println!("\n{records} records\n");
    println!(
        "{:<16} {:<7} {:>7} {:>9}  and what else it says",
        "question", "type", "scored", "accuracy"
    );
    for (id, tally) in tallies {
        let accuracy = tally.hits as f64 / tally.scored.max(1) as f64;
        let extra = match tally.kind {
            "noul" => format!(
                "p(yes) {} when yes, {} when no, AUC {}",
                number(mean(&tally.yes)),
                number(mean(&tally.no)),
                number(auc(&tally.yes, &tally.no))
            ),
            "score" => format!("mean absolute error {} levels", number(mean(&tally.errors))),
            // A choice's own confidence, split by whether it was right: the
            // cheapest look at whether the number means anything.
            _ => {
                let split = |want: bool| {
                    mean(
                        &tally
                            .confidence
                            .iter()
                            .filter(|(_, hit)| *hit == want)
                            .map(|(confidence, _)| *confidence)
                            .collect::<Vec<_>>(),
                    )
                };
                format!(
                    "confidence {} when right, {} when wrong",
                    number(split(true)),
                    number(split(false))
                )
            }
        };
        println!(
            "{id:<16} {:<7} {:>7} {:>8.1}%  {extra}",
            tally.kind,
            tally.scored,
            accuracy * 100.0
        );
    }

    // The confusion, where the options fit on a line.
    for (id, tally) in tallies {
        if tally.confusion.is_empty() || tally.confusion.len() > 8 {
            continue;
        }
        println!("\n{id}: label -> what it answered");
        for (label, answers) in &tally.confusion {
            let mut shown: Vec<String> = answers
                .iter()
                .map(|(answer, count)| {
                    format!(
                        "{count} {answer}{}",
                        if answer == label { " ok" } else { "" }
                    )
                })
                .collect();
            shown.sort();
            println!("  {label:<14} {}", shown.join(", "));
        }
    }

    // What the distribution is for: where a threshold could sit.
    println!("\naccuracy by confidence, which is where a routing threshold comes from");
    println!(
        "{:<16} {:<12} {:>6} {:>9}",
        "question", "confidence", "n", "accuracy"
    );
    for (id, tally) in tallies {
        for (low, high) in [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)] {
            let bucket: Vec<bool> = tally
                .confidence
                .iter()
                .filter(|(confidence, _)| *confidence >= low && *confidence < high)
                .map(|(_, hit)| *hit)
                .collect();
            if bucket.is_empty() {
                continue;
            }
            let hits = bucket.iter().filter(|hit| **hit).count();
            println!(
                "{id:<16} {:<12} {:>6} {:>8.1}%",
                format!("{low:.1}-{:.1}", high.min(1.0)),
                bucket.len(),
                hits as f64 / bucket.len() as f64 * 100.0
            );
        }
    }

    if errors == 0 {
        return;
    }
    println!("\nthe most confident mistakes, which is usually where a question's wording is");
    for (id, tally) in tallies {
        let mut misses = tally.misses.clone();
        misses.sort_by(|a, b| b.0.total_cmp(&a.0));
        for (confidence, line, label, answered) in misses.into_iter().take(errors) {
            println!("{id:<16} line {line:<5} label {label:<14} answered {answered:<14} at {confidence:.2}");
        }
    }
}

fn as_json(tallies: &IndexMap<String, Tally>, records: usize) -> String {
    let mut out = format!("{{\"records\": {records}, \"questions\": {{");
    for (index, (id, tally)) in tallies.iter().enumerate() {
        if index > 0 {
            out.push_str(", ");
        }
        out.push_str(&format!(
            "{}: {{\"type\": {}, \"scored\": {}, \"hits\": {}, \"accuracy\": {:.4}",
            serde_json::to_string(id).unwrap_or_default(),
            serde_json::to_string(tally.kind).unwrap_or_default(),
            tally.scored,
            tally.hits,
            tally.hits as f64 / tally.scored.max(1) as f64,
        ));
        if tally.kind == "noul" {
            out.push_str(&format!(
                ", \"auc\": {}, \"mean_p_when_yes\": {}, \"mean_p_when_no\": {}",
                json_number(auc(&tally.yes, &tally.no)),
                json_number(mean(&tally.yes)),
                json_number(mean(&tally.no))
            ));
        }
        if tally.kind == "score" {
            out.push_str(&format!(
                ", \"mean_absolute_error\": {}",
                json_number(mean(&tally.errors))
            ));
        }
        out.push('}');
    }
    out.push_str("}}");
    out
}

/// A measure with nothing behind it — a class with no records, say — is a dash
/// rather than `NaN`.
fn number(value: f64) -> String {
    if value.is_nan() {
        return String::from("—");
    }
    format!("{value:.2}")
}

/// The same for JSON, where `NaN` is not a value at all.
fn json_number(value: f64) -> String {
    if value.is_nan() {
        return String::from("null");
    }
    format!("{value:.4}")
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// The probability that a positive record scores above a negative one, ties
/// counted half: what a yes/no answer is worth before any threshold is chosen.
fn auc(yes: &[f64], no: &[f64]) -> f64 {
    if yes.is_empty() || no.is_empty() {
        return f64::NAN;
    }
    let mut better = 0.0;
    for p in yes {
        for q in no {
            better += match p.total_cmp(q) {
                std::cmp::Ordering::Greater => 1.0,
                std::cmp::Ordering::Equal => 0.5,
                std::cmp::Ordering::Less => 0.0,
            };
        }
    }
    better / (yes.len() * no.len()) as f64
}

// ---------------------------------------------------------------------------
// Records
// ---------------------------------------------------------------------------

fn load(
    path: &std::path::Path,
    questions: &IndexMap<String, Question>,
    limit: Option<usize>,
) -> Result<Vec<Record>, Box<dyn std::error::Error>> {
    let text = read(path)?;
    let mut records = Vec::new();
    for (index, line) in text.lines().enumerate() {
        let line_number = index + 1;
        if line.trim().is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(line)
            .map_err(|e| format!("{}:{line_number}: {e}", path.display()))?;
        let state = value
            .get("state")
            .cloned()
            .ok_or_else(|| format!("{}:{line_number}: no \"state\"", path.display()))?;
        let labels: IndexMap<String, Value> = match value.get("labels") {
            Some(Value::Object(map)) => map.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
            Some(_) => {
                return Err(format!(
                    "{}:{line_number}: \"labels\" is not an object",
                    path.display()
                )
                .into())
            }
            None => IndexMap::new(),
        };
        // A label for a question that does not exist would otherwise be a perfect
        // score on nothing — the commonest way to fool yourself with an eval set.
        for id in labels.keys() {
            if !questions.contains_key(id) {
                return Err(format!(
                    "{}:{line_number}: labelled {id:?}, which is not one of the questions ({})",
                    path.display(),
                    questions.keys().cloned().collect::<Vec<_>>().join(", ")
                )
                .into());
            }
        }
        records.push(Record {
            line: line_number,
            state,
            labels,
        });
        if limit.is_some_and(|limit| records.len() >= limit) {
            break;
        }
    }
    Ok(records)
}

/// A `noul` label: `true`/`false`, or the words for them.
fn truthy(label: &Value) -> bool {
    match label {
        Value::Bool(value) => *value,
        Value::Number(number) => number.as_f64().unwrap_or(0.0) > 0.5,
        Value::String(text) => matches!(
            text.to_lowercase().as_str(),
            "true" | "yes" | "ja" | "y" | "1"
        ),
        _ => false,
    }
}

/// A `score` label: the level index, or the level's own description.
fn level(label: &Value, legend: &IndexMap<String, String>) -> Option<i64> {
    match label {
        Value::Number(number) => number.as_i64(),
        Value::String(text) => legend
            .iter()
            .find(|(_, description)| description.eq_ignore_ascii_case(text))
            .and_then(|(index, _)| index.parse().ok()),
        _ => None,
    }
}

fn read(path: &std::path::Path) -> std::io::Result<String> {
    if path == std::path::Path::new("-") {
        let mut text = String::new();
        std::io::Read::read_to_string(&mut std::io::stdin(), &mut text)?;
        return Ok(text);
    }
    std::fs::read_to_string(path)
}

const USAGE: &str = "\
usage: eval --base <dir> [--checkpoint <dir>] --records <file.jsonl>
            (--questions <file.json> | --request <file.json>)

  --head <file>     the pointer head, if not <checkpoint>/head.pt
  --records <file>  one JSON object per line: \"state\" and \"labels\" (or - for stdin)
  --questions       the questions map, as in a System One request
  --request         take the questions out of a whole request instead
  --batch <n>       records answered in one prefill pass (default 1)
  --limit <n>       stop after n records
  --errors <n>      print the n most confident mistakes per question (default 5)
  --json            the tallies as JSON instead of a report

tests/eval/README.md has the record format and what the numbers mean.";

fn parse() -> Result<Options, String> {
    let mut base = None;
    let mut records = None;
    let mut options = Options {
        base: PathBuf::new(),
        checkpoint: None,
        head: None,
        questions: None,
        request: None,
        records: PathBuf::new(),
        batch: 1,
        limit: None,
        errors: 5,
        json: false,
    };

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
            "--questions" => options.questions = Some(PathBuf::from(value()?)),
            "--request" => options.request = Some(PathBuf::from(value()?)),
            "--records" => records = Some(PathBuf::from(value()?)),
            "--batch" => options.batch = value()?.parse().map_err(|e| format!("--batch: {e}"))?,
            "--limit" => {
                options.limit = Some(value()?.parse().map_err(|e| format!("--limit: {e}"))?)
            }
            "--errors" => {
                options.errors = value()?.parse().map_err(|e| format!("--errors: {e}"))?
            }
            "--json" => options.json = true,
            "-h" | "--help" => return Err(String::from("eval")),
            other => return Err(format!("unknown argument {other}")),
        }
    }
    options.base = base.ok_or("--base is required")?;
    options.records = records.ok_or("--records is required")?;
    Ok(options)
}
