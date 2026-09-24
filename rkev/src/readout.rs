//! The readout: hidden states in, answers out.
//!
//! Kev generates nothing. A checkpoint is a LoRA adapter plus a small pointer
//! head, and the head is the whole output layer: it scores each option's
//! `</opt>` hidden state against its question's `<decide>` hidden state, and a
//! softmax over those scores is the answer. Mirrors `kev/model.py`'s
//! `PointerHead` and `kev/api.py`'s `to_answers`.
//!
//! Everything here is plain `f32`/`f64` arithmetic on hidden states someone else
//! produced, which is why it can be tested without a model.

use indexmap::IndexMap;

use crate::error::{Error, Result};
use crate::prompt::{Kind, Plan};
use crate::types::Answer;

/// A `torch.nn.Linear`: `y = W x + b`, with `W` stored row-major as
/// `[outputs][inputs]`.
#[derive(Debug, Clone)]
pub struct Linear {
    weight: Vec<f32>,
    bias: Vec<f32>,
    inputs: usize,
}

impl Linear {
    /// `weight` is `outputs * inputs` values, row-major; `bias` is one value
    /// per output, as the checkpoint stores them.
    pub fn new(weight: Vec<f32>, bias: Vec<f32>, inputs: usize) -> Result<Self> {
        if inputs == 0 || bias.is_empty() || weight.len() != bias.len() * inputs {
            return Err(Error::Engine(format!(
                "a {}x{inputs} linear needs {} weights and {} biases, got {} and {}",
                bias.len(),
                bias.len() * inputs,
                bias.len(),
                weight.len(),
                bias.len(),
            )));
        }
        Ok(Self {
            weight,
            bias,
            inputs,
        })
    }

    fn outputs(&self) -> usize {
        self.bias.len()
    }

    fn apply(&self, x: &[f32]) -> Vec<f32> {
        self.weight
            .chunks(self.inputs)
            .zip(&self.bias)
            .map(|(row, bias)| bias + row.iter().zip(x).map(|(w, x)| w * x).sum::<f32>())
            .collect()
    }
}

/// The pointer head: the output layer of every Kev checkpoint.
///
/// Two projections into a shared pointer space, and a dot product per option.
/// The scale is `1/sqrt(pointer dimension)`, as in the Python head.
#[derive(Debug, Clone)]
pub struct PointerHead {
    query: Linear,
    key: Linear,
    scale: f32,
    temperature: f32,
}

impl PointerHead {
    /// The head from a checkpoint's two projections: `query` reads `<decide>`,
    /// `key` reads each `</opt>`.
    pub fn new(query: Linear, key: Linear) -> Result<Self> {
        if query.outputs() != key.outputs() || query.inputs != key.inputs {
            return Err(Error::Engine(format!(
                "the pointer head's projections disagree: query is {}x{}, key is {}x{}",
                query.outputs(),
                query.inputs,
                key.outputs(),
                key.inputs,
            )));
        }
        let scale = 1.0 / (query.outputs() as f32).sqrt();
        Ok(Self {
            query,
            key,
            scale,
            temperature: 1.0,
        })
    }

    /// Divide the scores by `temperature`, the calibration a checkpoint carries
    /// (about 2.1 to 2.4 for the released models; `1.0` is the raw head).
    ///
    /// It never changes which option wins, only how confident the distribution
    /// is.
    pub fn with_temperature(mut self, temperature: f32) -> Result<Self> {
        if temperature.is_nan() || temperature <= 0.0 {
            return Err(Error::Engine(format!(
                "the pointer head's temperature must be positive, got {temperature}"
            )));
        }
        self.temperature = temperature;
        Ok(self)
    }

    /// The same head with its two projections exchanged — deliberately the wrong
    /// way round.
    ///
    /// Which projection reads `<decide>` and which reads `</opt>` comes from the
    /// reference's own naming (`q` and `k` in `head.pt`), and swapping them yields
    /// a different but perfectly plausible distribution: no shape check and no
    /// self-consistency test can tell the two apart, because both sides of every
    /// comparison would be swapped alike.
    ///
    /// What *can* tell them apart is a trained checkpoint. A head only scores the
    /// right option highly in the orientation it was trained in, so answering
    /// questions whose answer is not in doubt with this head and with the ordinary
    /// one separates them — that is what `examples/sanity.rs` does with it, and it
    /// needs no reference numbers to do it.
    pub fn swapped(self) -> Self {
        Self {
            query: self.key,
            key: self.query,
            ..self
        }
    }

    /// The hidden size this head expects from the backbone.
    pub fn hidden_size(&self) -> usize {
        self.query.inputs
    }

    /// One logit per option, from the `<decide>` and `</opt>` hidden states.
    pub fn logits(&self, decide: &[f32], options: &[Vec<f32>]) -> Result<Vec<f32>> {
        let expected = self.hidden_size();
        let wrong = std::iter::once(decide.len())
            .chain(options.iter().map(Vec::len))
            .any(|len| len != expected);
        if wrong {
            return Err(Error::Engine(format!(
                "the pointer head takes {expected}-dimensional hidden states, \
                 the backend returned other widths"
            )));
        }

        let query = self.query.apply(decide);
        Ok(options
            .iter()
            .map(|option| {
                let key = self.key.apply(option);
                let score: f32 = key.iter().zip(&query).map(|(k, q)| k * q).sum();
                score * self.scale / self.temperature
            })
            .collect())
    }
}

/// The distribution over one question's options.
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|z| (z - max).exp()).collect();
    let total: f32 = exp.iter().sum();
    exp.iter().map(|e| e / total).collect()
}

/// Serialisation precision for probabilities and the scalars derived from them.
///
/// Four decimals keeps a rounded 255-option distribution within TypeSafe's
/// `|sum - 1| < 0.02` tolerance.
fn round_prob(x: f64) -> f64 {
    (x * 10_000.0).round() / 10_000.0
}

/// `(p_max - 1/K) / (1 - 1/K)`: how far the winner is above a uniform guess.
/// A single option is certain by construction.
fn choice_confidence(p: &[f64]) -> f64 {
    let k = p.len() as f64;
    if p.len() == 1 {
        return 1.0;
    }
    let max = p.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    (max - 1.0 / k) / (1.0 - 1.0 / k)
}

/// `1 - E|level - mode| / (L - 1)`: how tightly a score sits on its most likely
/// level. An approximation of TypeSafe's unpublished statistic.
fn score_confidence(p: &[f64]) -> f64 {
    if p.len() == 1 {
        return 1.0;
    }
    let mode = argmax(p) as f64;
    let spread: f64 = p
        .iter()
        .enumerate()
        .map(|(level, p)| p * (level as f64 - mode).abs())
        .sum();
    1.0 - spread / (p.len() - 1) as f64
}

/// The first index holding the largest value, as Python's `max(range(n), key=)`
/// resolves a tie.
fn argmax(p: &[f64]) -> usize {
    let mut best = 0;
    for (index, value) in p.iter().enumerate() {
        if *value > p[best] {
            best = index;
        }
    }
    best
}

/// Read one question's distribution back as the answer its type calls for.
pub(crate) fn answer(probabilities: &[f32], plan: &Plan) -> Result<Answer> {
    if probabilities.len() != plan.keys.len() {
        return Err(Error::Engine(format!(
            "question {:?} has {} options but the readout produced {} probabilities",
            plan.id,
            plan.keys.len(),
            probabilities.len()
        )));
    }
    let p: Vec<f64> = probabilities.iter().map(|p| *p as f64).collect();
    let distribution = || -> IndexMap<String, f64> {
        plan.keys
            .iter()
            .cloned()
            .zip(p.iter().map(|p| round_prob(*p)))
            .collect()
    };

    Ok(match plan.kind {
        // Two options, [no, yes]; the probability of yes is the answer, and
        // there is no confidence to report because the probability is one.
        Kind::Noul => Answer::Noul {
            noul: round_prob(p[1]),
        },
        Kind::Choice => Answer::Choice {
            choice: plan.keys[argmax(&p)].clone(),
            confidence: round_prob(choice_confidence(&p)),
            probabilities: distribution(),
        },
        // The score is the distribution's mean level index, so a bimodal
        // answer lands between its two levels - read it with the
        // probabilities, not on its own.
        Kind::Score => Answer::Score {
            score: round_prob(
                p.iter()
                    .enumerate()
                    .map(|(level, p)| level as f64 * p)
                    .sum(),
            ),
            confidence: round_prob(score_confidence(&p)),
            legend: plan.legend.clone().unwrap_or_default(),
            probabilities: distribution(),
        },
    })
}

/// The answers as the Python server serialises them.
///
/// This is what `usage.output_tokens` counts, which is why it exists and why it
/// is exact; it is also what a caller replacing the server has to emit.
///
/// `output_tokens` is a billing-style figure: the tokens of the serialised
/// answers, since Kev generates none. Matching it means matching `json.dumps`
/// exactly — its `", "` and `": "` separators, its `\uXXXX` escaping of
/// non-ASCII, its key order, and `repr` for floats.
pub fn answers_json(answers: &IndexMap<String, Answer>) -> String {
    let mut out = String::from("{");
    for (index, (id, answer)) in answers.iter().enumerate() {
        if index > 0 {
            out.push_str(", ");
        }
        push_string(&mut out, id);
        out.push_str(": ");
        push_answer(&mut out, answer);
    }
    out.push('}');
    out
}

fn push_answer(out: &mut String, answer: &Answer) {
    match answer {
        Answer::Noul { noul } => {
            out.push_str("{\"type\": \"noul\", \"noul\": ");
            push_float(out, *noul);
            out.push('}');
        }
        Answer::Choice {
            choice,
            confidence,
            probabilities,
        } => {
            out.push_str("{\"type\": \"choice\", \"choice\": ");
            push_string(out, choice);
            out.push_str(", \"confidence\": ");
            push_float(out, *confidence);
            out.push_str(", \"probabilities\": ");
            push_map(out, probabilities.iter().map(|(k, v)| (k, Cell::Float(*v))));
            out.push('}');
        }
        Answer::Score {
            score,
            confidence,
            legend,
            probabilities,
        } => {
            out.push_str("{\"type\": \"score\", \"score\": ");
            push_float(out, *score);
            out.push_str(", \"legend\": ");
            push_map(out, legend.iter().map(|(k, v)| (k, Cell::Text(v))));
            out.push_str(", \"probabilities\": ");
            push_map(out, probabilities.iter().map(|(k, v)| (k, Cell::Float(*v))));
            out.push_str(", \"confidence\": ");
            push_float(out, *confidence);
            out.push('}');
        }
    }
}

enum Cell<'a> {
    Float(f64),
    Text(&'a str),
}

fn push_map<'a, I>(out: &mut String, entries: I)
where
    I: Iterator<Item = (&'a String, Cell<'a>)>,
{
    out.push('{');
    for (index, (key, value)) in entries.enumerate() {
        if index > 0 {
            out.push_str(", ");
        }
        push_string(out, key);
        out.push_str(": ");
        match value {
            Cell::Float(f) => push_float(out, f),
            Cell::Text(t) => push_string(out, t),
        }
    }
    out.push('}');
}

/// A float the way Python's `repr` writes it: shortest round trip, but never
/// without a fractional part.
fn push_float(out: &mut String, value: f64) {
    let text = format!("{value}");
    out.push_str(&text);
    if !text.contains('.') && !text.contains('e') {
        out.push_str(".0");
    }
}

/// A string the way `json.dumps` writes it, non-ASCII included.
fn push_string(out: &mut String, value: &str) {
    out.push('"');
    for c in value.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c if c.is_ascii() => out.push(c),
            // ensure_ascii=True: astral characters go out as a surrogate pair.
            c => {
                let mut utf16 = [0u16; 2];
                for unit in c.encode_utf16(&mut utf16) {
                    out.push_str(&format!("\\u{unit:04x}"));
                }
            }
        }
    }
    out.push('"');
}
