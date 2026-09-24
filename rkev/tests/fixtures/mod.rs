//! What the tests build for themselves: checkpoints on disk, and a backend with
//! no model in it.
//!
//! Not every test uses every piece of this, and a test module is compiled per
//! test file, so the unused ones are allowed rather than split up.
#![allow(dead_code)]
//!
//!
//! Small enough to run in milliseconds and made of noise, so the numbers mean
//! nothing: what the backbone tests assert are the properties that hold for any
//! weights. The writers are hand-rolled rather than pulled in as
//! dev-dependencies — a safetensors file is a length, a JSON header and the
//! floats, and that is the whole format this needs.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use rkev::{
    Error, Forward, Linear, LocalEngine, OptionSlot, Pass, PointerHead, Result, DECIDE, OPTION,
    OPTION_END, QUESTION, STATE,
};

/// Words the toy tokenizer knows; everything else becomes `[UNK]`.
pub const WORDS: [&str; 16] = [
    "a", "ticket", "about", "money", "late", "shoes", "no", "yes", "returns", "billing", "calm",
    "angry", "is", "this", "?", "the",
];

pub fn vocab_size() -> usize {
    1 + 5 + WORDS.len()
}

/// Named tensors, in the shape and order a safetensors file wants them.
pub type Tensors = BTreeMap<String, (Vec<usize>, Vec<f32>)>;

/// Deterministic small values, so a test failure is always the code's fault.
pub struct Noise(pub u64);

impl Noise {
    pub fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.0 >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * 0.2
    }

    pub fn values(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next()).collect()
    }

    /// Norm weights sit around 1.0, as trained ones do.
    pub fn around_one(&mut self, n: usize) -> Vec<f32> {
        self.values(n).into_iter().map(|v| 1.0 + v).collect()
    }
}

/// The minimum a safetensors reader needs: a u64 header length, a JSON header,
/// then the f32s.
pub fn write_safetensors(path: &Path, tensors: &Tensors) {
    let mut header = String::from("{");
    let mut offset = 0;
    for (index, (name, (shape, values))) in tensors.iter().enumerate() {
        let end = offset + values.len() * 4;
        if index > 0 {
            header.push(',');
        }
        let shape = shape
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",");
        header.push_str(&format!(
            "\"{name}\":{{\"dtype\":\"F32\",\"shape\":[{shape}],\"data_offsets\":[{offset},{end}]}}"
        ));
        offset = end;
    }
    header.push('}');
    while (8 + header.len()) % 8 != 0 {
        header.push(' ');
    }

    let mut bytes = Vec::new();
    bytes.extend((header.len() as u64).to_le_bytes());
    bytes.extend(header.as_bytes());
    for (_, values) in tensors.values() {
        for value in values {
            bytes.extend(value.to_le_bytes());
        }
    }
    fs::write(path, bytes).unwrap();
}

/// A tokenizer in the shape Qwen ships one: the five delimiters in the
/// vocabulary *and* as added tokens, which is why an added token can be matched
/// inside ordinary text and why caller text has to be escaped.
pub fn tokenizer_json() -> String {
    let delimiters = [STATE, QUESTION, OPTION, OPTION_END, DECIDE];
    let mut vocab = vec![(String::from("[UNK]"), 0)];
    for (index, token) in delimiters.iter().enumerate() {
        vocab.push((token.to_string(), index + 1));
    }
    for (index, word) in WORDS.iter().enumerate() {
        vocab.push((word.to_string(), index + 6));
    }
    let entries: Vec<String> = vocab
        .iter()
        .map(|(token, id)| format!("\"{token}\":{id}"))
        .collect();
    let added: Vec<String> = delimiters
        .iter()
        .enumerate()
        .map(|(index, token)| {
            format!(
                r#"{{"id":{},"content":"{token}","single_word":false,"lstrip":false,
                     "rstrip":false,"normalized":false,"special":true}}"#,
                index + 1
            )
        })
        .collect();
    format!(
        r#"{{"version":"1.0","truncation":null,"padding":null,"added_tokens":[{}],
            "normalizer":null,"pre_tokenizer":{{"type":"Whitespace"}},"post_processor":null,
            "decoder":null,
            "model":{{"type":"WordLevel","vocab":{{{}}},"unk_token":"[UNK]"}}}}"#,
        added.join(","),
        entries.join(",")
    )
}

/// A pointer head as safetensors, which carries no temperature and so leaves the
/// head raw.
pub fn write_head(dir: &Path, hidden: usize, pointer: usize) {
    let mut noise = Noise(99);
    let mut tensors = Tensors::new();
    for name in ["q", "k"] {
        tensors.insert(
            format!("head.{name}.weight"),
            (vec![pointer, hidden], noise.values(pointer * hidden)),
        );
        tensors.insert(
            format!("head.{name}.bias"),
            (vec![pointer], noise.values(pointer)),
        );
    }
    write_safetensors(&dir.join("head.safetensors"), &tensors);
}

/// One fixed directory per test, rebuilt on every run rather than piling up in
/// the temp directory.
pub fn fresh_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("kev-{name}"));
    fs::remove_dir_all(&dir).ok();
    fs::create_dir_all(&dir).unwrap();
    dir
}

// ---------------------------------------------------------------------------
// A backend with no model in it
// ---------------------------------------------------------------------------

/// The ids this stub's tokenizer gives Kev's five delimiters. Caller text
/// tokenises one byte per token, offset well clear of these.
pub const STATE_ID: u32 = 1;
pub const QUESTION_ID: u32 = 2;
pub const OPTION_ID: u32 = 3;
pub const OPTION_END_ID: u32 = 4;
pub const DECIDE_ID: u32 = 5;
pub const TEXT_BASE: u32 = 1000;

/// What the engine asked the backend to do.
#[derive(Default)]
pub struct Log {
    /// Every piece of text tokenised, in order.
    pub texts: Vec<String>,
    pub passes: Vec<Recorded>,
}

pub struct Recorded {
    pub ids: Vec<u32>,
    pub positions: Vec<u32>,
    pub segments: Vec<u32>,
    pub readout: Vec<usize>,
    pub options: Vec<OptionSlot>,
}

impl Recorded {
    pub fn as_pass(&self) -> Pass<'_> {
        Pass::new(&self.ids, &self.positions, &self.segments, &self.readout)
            .with_options(&self.options)
    }

    /// The token indices belonging to question `question` (1-based).
    pub fn branch(&self, question: u32) -> Vec<usize> {
        (0..self.ids.len())
            .filter(|index| self.segments[*index] == question)
            .collect()
    }
}

pub struct Stub {
    /// One target distribution per question, in the order the questions are
    /// asked; consumed as the engine works through them.
    pub targets: Vec<Vec<f64>>,
    pub served: usize,
    /// Answer through `Pass::rows`, the way a backbone that cannot honour the
    /// block-causal mask has to.
    pub rows: bool,
    pub log: Arc<Mutex<Log>>,
    pub passes: Arc<AtomicUsize>,
}

impl Stub {
    pub fn new(targets: Vec<Vec<f64>>) -> (Self, Arc<Mutex<Log>>, Arc<AtomicUsize>) {
        let log = Arc::new(Mutex::new(Log::default()));
        let passes = Arc::new(AtomicUsize::new(0));
        let stub = Stub {
            targets,
            served: 0,
            rows: false,
            log: Arc::clone(&log),
            passes: Arc::clone(&passes),
        };
        (stub, log, passes)
    }

    /// Hidden states for one pass, without recording it: a `<decide>` position
    /// gets `[sqrt(2), 0]`, an option's `</opt>` gets `[ln p, 0]`. With the
    /// identity pointer head below that makes the head's logit for an option
    /// exactly `ln p`, so the softmax is the target distribution.
    fn states(&mut self, pass: &Pass<'_>) -> Result<Vec<Vec<f32>>> {
        let mut states = Vec::with_capacity(pass.readout.len());
        let mut option = 0;
        for position in pass.readout {
            match pass.ids[*position] {
                DECIDE_ID => {
                    self.served += 1;
                    option = 0;
                    states.push(vec![2f32.sqrt(), 0.0]);
                }
                OPTION_END_ID => {
                    let target = self.targets.get(self.served - 1).ok_or_else(|| {
                        Error::Engine(format!("stub has no target for question {}", self.served))
                    })?;
                    states.push(vec![target[option].ln() as f32, 0.0]);
                    option += 1;
                }
                id => {
                    return Err(Error::Engine(format!(
                        "the readout asked for position {position}, which holds {id}, \
                         not a <decide> or </opt> token"
                    )))
                }
            }
        }
        Ok(states)
    }
}

impl Forward for Stub {
    fn tokenise(&mut self, text: &str) -> Result<Vec<u32>> {
        self.log.lock().unwrap().texts.push(text.to_string());
        Ok(text.bytes().map(|b| TEXT_BASE + u32::from(b)).collect())
    }

    fn delimiter(&mut self, token: &str) -> Result<u32> {
        match token {
            STATE => Ok(STATE_ID),
            QUESTION => Ok(QUESTION_ID),
            OPTION => Ok(OPTION_ID),
            OPTION_END => Ok(OPTION_END_ID),
            DECIDE => Ok(DECIDE_ID),
            other => Err(Error::Engine(format!("unknown delimiter {other}"))),
        }
    }

    fn hidden(&mut self, pass: &Pass<'_>) -> Result<Vec<Vec<f32>>> {
        self.passes.fetch_add(1, Ordering::SeqCst);
        self.log.lock().unwrap().passes.push(Recorded {
            ids: pass.ids.to_vec(),
            positions: pass.positions.to_vec(),
            segments: pass.segments.to_vec(),
            readout: pass.readout.to_vec(),
            options: pass.options.to_vec(),
        });

        if !self.rows {
            return self.states(pass);
        }
        // One independent causal row per question, concatenated in order -
        // what a Gated DeltaNet backbone is left with.
        let mut states = Vec::new();
        for row in pass.rows() {
            states.extend(self.states(&row.as_pass())?);
        }
        Ok(states)
    }
}

/// The identity pointer head: `logit(option) = h_opt . h_decide / sqrt(2)`.
/// A real one carries a checkpoint's trained projections.
pub fn identity_head() -> PointerHead {
    let projection = || Linear::new(vec![1.0, 0.0, 0.0, 1.0], vec![0.0, 0.0], 2).unwrap();
    PointerHead::new(projection(), projection()).unwrap()
}

pub fn engine(targets: Vec<Vec<f64>>) -> (LocalEngine, Arc<Mutex<Log>>, Arc<AtomicUsize>) {
    let (stub, log, passes) = Stub::new(targets);
    (LocalEngine::new(stub, identity_head()), log, passes)
}
