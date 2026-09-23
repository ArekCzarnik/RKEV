//! Checkpoints the tests write themselves.
//!
//! Small enough to run in milliseconds and made of noise, so the numbers mean
//! nothing: what the backbone tests assert are the properties that hold for any
//! weights. The writers are hand-rolled rather than pulled in as
//! dev-dependencies — a safetensors file is a length, a JSON header and the
//! floats, and that is the whole format this needs.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use kev_client::{DECIDE, OPTION, OPTION_END, QUESTION, STATE};

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
