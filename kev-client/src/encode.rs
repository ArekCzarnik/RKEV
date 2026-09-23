//! The token layout of one request: `<state>`, then one branch per question.
//!
//! Mirrors `kev/model.py` (`SPECIAL`, `user_tokens`, `encode`, `rows_of`,
//! `branch_mask`). The layout is the whole interface to the model:
//!
//! ```text
//! <state> ...state...
//! <q> instructions <opt> option 1 </opt> <opt> option 2 </opt> ... <decide>
//! <q> instructions <opt> option 1 </opt> ... <decide>
//! ```
//!
//! A token may read the state and its own question, nothing else — that is
//! what keeps the questions from reading each other, and it is why a backend
//! needs [`Pass::attends`](crate::Pass) rather than a plain causal mask. Each branch's
//! position ids restart just after the state, so the state is processed once
//! however many questions come with it.

use crate::error::{Error, Result};
use crate::prompt::Record;

/// Kev's five delimiters, in layout order: state, question, option,
/// end-of-option, decide.
///
/// They are rarely-used Qwen special tokens rather than new ones, so no
/// embedding rows have to be added — the LoRA adapter gives them their meaning.
/// A backend must map these to the ids of its own tokenizer.
pub const STATE: &str = "<|fim_prefix|>";
/// Opens one question's branch. See [`STATE`].
pub const QUESTION: &str = "<|fim_middle|>";
/// Opens one option span. See [`STATE`].
pub const OPTION: &str = "<|box_start|>";
/// Closes one option span; its hidden state is what the pointer head scores.
pub const OPTION_END: &str = "<|box_end|>";
/// The position the pointer head compares the options against.
pub const DECIDE: &str = "<|fim_suffix|>";

/// All five delimiters in layout order.
pub const SPECIAL: [&str; 5] = [STATE, QUESTION, OPTION, OPTION_END, DECIDE];

/// The token ids a backend's tokenizer uses for [`SPECIAL`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct Delimiters {
    pub state: u32,
    pub question: u32,
    pub option: u32,
    pub option_end: u32,
    pub decide: u32,
}

/// How many tokens a state and a single question branch may take.
///
/// [`Limits::serving`] is what `kev.serve` uses. Training used 384 state tokens
/// and 1024 for state plus one question, so longer inputs are allowed but were
/// never trained on.
#[derive(Debug, Clone, Copy)]
pub struct Limits {
    /// The state is truncated to this many tokens, delimiter included.
    pub max_state: usize,
    /// A state plus one question branch may not exceed this.
    pub max_branch: usize,
}

impl Limits {
    /// The serving limits: 8192 tokens for the state, 8192 for state plus one
    /// question branch.
    pub fn serving() -> Self {
        Self {
            max_state: 8192,
            max_branch: 8192,
        }
    }

    /// The context the released checkpoints were trained on: 384 state tokens,
    /// 1024 for state plus one branch.
    pub fn training() -> Self {
        Self {
            max_state: 384,
            max_branch: 1024,
        }
    }
}

/// One packed request: the tokens, where they sit, and where to read from.
#[derive(Debug, Clone)]
pub(crate) struct Encoding {
    /// The token ids, state first, then one branch per question.
    pub ids: Vec<u32>,
    /// `0` for the state, `k` for the tokens of question `k` (1-based).
    pub segments: Vec<u32>,
    /// Position ids. Each branch restarts just after the state.
    pub positions: Vec<u32>,
    /// Index of each question's `<decide>` token.
    pub decide: Vec<usize>,
    /// Index of each option's `</opt>` token, per question, in option order.
    pub options: Vec<Vec<usize>>,
}

impl Encoding {
    /// The positions whose hidden states the readout needs, in the order the
    /// pointer head wants them: each question's `<decide>` first, then its
    /// options.
    pub fn readout(&self) -> Vec<usize> {
        let mut positions = Vec::new();
        for (decide, options) in self.decide.iter().zip(&self.options) {
            positions.push(*decide);
            positions.extend(options.iter().copied());
        }
        positions
    }

    /// How many readout positions one question contributes: `<decide>` plus
    /// one per option.
    pub fn readout_width(&self, question: usize) -> usize {
        1 + self.options[question].len()
    }
}

/// Rewrite anything that looks like a special token so caller text cannot
/// produce one.
///
/// `<|name|>` becomes `<¦name¦>`, as in `kev.model.user_tokens`: option and
/// question boundaries have to be unforgeable, or a state could pretend to be
/// a question.
fn escape_specials(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < bytes.len() {
        if let Some(name) = special_at(bytes, i) {
            out.push_str("<\u{a6}");
            out.push_str(name);
            out.push_str("\u{a6}>");
            i += name.len() + 4;
        } else {
            let start = i;
            i += 1;
            while i < bytes.len() && !text.is_char_boundary(i) {
                i += 1;
            }
            out.push_str(&text[start..i]);
        }
    }
    out
}

/// The `name` of a `<|name|>` starting at `at`, if there is one.
fn special_at(bytes: &[u8], at: usize) -> Option<&str> {
    if bytes.get(at) != Some(&b'<') || bytes.get(at + 1) != Some(&b'|') {
        return None;
    }
    let start = at + 2;
    let mut end = start;
    while matches!(bytes.get(end), Some(c) if c.is_ascii_alphanumeric() || *c == b'_') {
        end += 1;
    }
    if end == start || bytes.get(end) != Some(&b'|') || bytes.get(end + 1) != Some(&b'>') {
        return None;
    }
    // ASCII by construction, so the slice is valid UTF-8.
    std::str::from_utf8(&bytes[start..end]).ok()
}

/// Pack one record into tokens, positions, segments and readout indices.
///
/// `tokenise` must not emit special tokens — the text is escaped first, so a
/// tokenizer that treats `<|...|>` as literal text is enough.
///
/// The state is truncated to fit `limits`, as the server does; a question whose
/// branch does not fit is a request error, not a silently shortened prompt.
pub(crate) fn encode<F>(
    record: &Record,
    delimiters: &Delimiters,
    limits: Limits,
    mut tokenise: F,
) -> Result<Encoding>
where
    F: FnMut(&str) -> Result<Vec<u32>>,
{
    if limits.max_state < 2 || limits.max_branch < limits.max_state {
        return Err(Error::Engine(format!(
            "a context of {} state tokens and {} per row is not usable",
            limits.max_state, limits.max_branch
        )));
    }

    let state = tokenise(&escape_specials(&record.state))?;
    let mut ids = Vec::with_capacity(state.len() + 1);
    ids.push(delimiters.state);
    ids.extend(state.iter().take(limits.max_state - 1));
    let state_len = ids.len();

    let mut segments = vec![0; state_len];
    let mut positions: Vec<u32> = (0..state_len as u32).collect();
    let mut decide = Vec::with_capacity(record.questions.len());
    let mut options = Vec::with_capacity(record.questions.len());

    for (index, question) in record.questions.iter().enumerate() {
        let segment = index as u32 + 1;
        let mut branch = vec![delimiters.question];
        branch.extend(tokenise(&escape_specials(&question.instructions))?);

        let mut ends = Vec::with_capacity(question.options.len());
        for option in &question.options {
            branch.push(delimiters.option);
            branch.extend(tokenise(&escape_specials(option))?);
            branch.push(delimiters.option_end);
            // The `</opt>` token just pushed: the hidden state the pointer
            // head reads this option from.
            ends.push(branch.len() - 1);
        }
        branch.push(delimiters.decide);

        if state_len + branch.len() > limits.max_branch {
            return Err(Error::ContextOverflow(format!(
                "branch too long: {} tokens with a {state_len}-token state (row limit {})",
                branch.len(),
                limits.max_branch
            )));
        }

        let base = ids.len();
        // Positions restart just after the state, so every question is laid
        // out as if it were the only one.
        positions.extend((state_len..state_len + branch.len()).map(|p| p as u32));
        segments.extend(std::iter::repeat(segment).take(branch.len()));
        decide.push(base + branch.len() - 1);
        options.push(ends.into_iter().map(|end| base + end).collect());
        ids.extend(branch);
    }

    Ok(Encoding {
        ids,
        segments,
        positions,
        decide,
        options,
    })
}
