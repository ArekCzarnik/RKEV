//! A local inference engine: the model runs in this process, no HTTP.
//!
//! [`LocalEngine`] owns everything Kev-specific — the prompt layout
//! ([`crate::encode`]) and the pointer-head readout ([`crate::readout`]) — and
//! leaves exactly one job to an inference backend behind [`Forward`]: run the
//! backbone and hand back hidden states. That split is deliberate. Kev's
//! checkpoints are a LoRA adapter plus a pointer head over a Qwen base, and the
//! head reads *hidden states*, never vocabulary logits, so a backend needs to
//! expose the backbone rather than a text-generation API.
//!
//! See `.claude/tasks/2026-09-23-local-inference.md` for which engines can do
//! that.

use std::fmt;
use std::future::Future;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use indexmap::IndexMap;

use crate::encode::{self, Delimiters, Encoding, Limits, SPECIAL};
use crate::error::{Error, Result};
use crate::prompt::{self, Plan, Record};
use crate::readout::{self, PointerHead};
use crate::system_one::SystemOne;
use crate::types::{Answer, SystemOneRequest, SystemOneResponse, Usage};

/// One forward pass, described to a backend.
///
/// The token ids, where each token sits, which question it belongs to, and
/// which positions the readout will want back. `positions` is not `0..n`: each
/// question's branch restarts just after the state, which is how one state
/// serves every question.
#[derive(Debug, Clone, Copy)]
pub struct Pass<'a> {
    /// Token ids to run.
    pub ids: &'a [u32],
    /// Position id per token.
    pub positions: &'a [u32],
    /// `0` for state tokens, `k` for the tokens of question `k`.
    pub segments: &'a [u32],
    /// The positions whose hidden states to return, in order.
    pub readout: &'a [usize],
}

impl Pass<'_> {
    /// Whether the token at `query` may read the token at `key`.
    ///
    /// Causal *and* blind across questions. Questions sharing one pass must not
    /// see each other, so this is not a plain causal mask: a backend that
    /// cannot express it must run [`Pass::rows`] instead.
    pub fn attends(&self, query: usize, key: usize) -> bool {
        key <= query && (self.segments[key] == 0 || self.segments[key] == self.segments[query])
    }

    /// `true` when a plain causal mask is enough — one question, or none.
    pub fn is_causal(&self) -> bool {
        self.segments.iter().all(|segment| *segment <= 1)
    }

    /// The same work as one independent causal row per question: the state
    /// tokens followed by that question's branch, at the positions they already
    /// carry.
    ///
    /// For backbones whose layers ignore attention masks — Qwen3.5's Gated
    /// DeltaNet layers, and so every current Kev checkpoint — this is the only
    /// exact form, and it is what the Python model runs there. Concatenating
    /// the rows' hidden states in order yields exactly what [`Forward::hidden`]
    /// must return — as long as the readout stays inside the branches, which is
    /// where `<decide>` and `</opt>` always are. A readout position in the state
    /// belongs to no row and is dropped, which the engine notices as a count
    /// mismatch rather than a wrong answer.
    /// The state tokens on their own: the part every question shares, and the
    /// part worth computing once.
    ///
    /// Together with [`Pass::branches`] this is [`Pass::rows`] taken apart, for a
    /// backend that can run the state once and continue from it.
    pub fn state(&self) -> OwnedPass {
        let state: Vec<usize> = (0..self.ids.len())
            .filter(|index| self.segments[*index] == 0)
            .collect();
        OwnedPass {
            ids: state.iter().map(|i| self.ids[*i]).collect(),
            positions: state.iter().map(|i| self.positions[*i]).collect(),
            segments: vec![0; state.len()],
            readout: Vec::new(),
        }
    }

    /// One pass per question, holding that question's branch tokens only, with
    /// the readout positions relative to the branch.
    ///
    /// A readout position in the state belongs to no branch and is dropped, as in
    /// [`Pass::rows`]; `<decide>` and `</opt>` are never there.
    pub fn branches(&self) -> Vec<OwnedPass> {
        let mut branches = Vec::new();
        let mut segment = 1;
        while self.segments.contains(&segment) {
            let indices: Vec<usize> = (0..self.ids.len())
                .filter(|index| self.segments[*index] == segment)
                .collect();
            let mut moved = vec![usize::MAX; self.ids.len()];
            for (offset, packed) in indices.iter().enumerate() {
                moved[*packed] = offset;
            }
            branches.push(OwnedPass {
                ids: indices.iter().map(|i| self.ids[*i]).collect(),
                positions: indices.iter().map(|i| self.positions[*i]).collect(),
                segments: vec![segment; indices.len()],
                readout: self
                    .readout
                    .iter()
                    .filter(|position| self.segments[**position] == segment)
                    .map(|position| moved[*position])
                    .collect(),
            });
            segment += 1;
        }
        branches
    }

    pub fn rows(&self) -> Vec<OwnedPass> {
        let state: Vec<usize> = (0..self.ids.len())
            .filter(|index| self.segments[*index] == 0)
            .collect();
        let mut rows = Vec::new();
        let mut segment = 1;
        while self.segments.contains(&segment) {
            let indices: Vec<usize> = state
                .iter()
                .copied()
                .chain((0..self.ids.len()).filter(|index| self.segments[*index] == segment))
                .collect();
            // Where each packed index ended up in this row, so the readout
            // positions can follow the tokens they point at.
            let mut moved = vec![usize::MAX; self.ids.len()];
            for (row_index, packed) in indices.iter().enumerate() {
                moved[*packed] = row_index;
            }
            rows.push(OwnedPass {
                ids: indices.iter().map(|i| self.ids[*i]).collect(),
                positions: indices.iter().map(|i| self.positions[*i]).collect(),
                segments: indices.iter().map(|i| self.segments[*i]).collect(),
                readout: self
                    .readout
                    .iter()
                    .filter(|position| self.segments[**position] == segment)
                    .map(|position| moved[*position])
                    .collect(),
            });
            segment += 1;
        }
        rows
    }
}

/// A [`Pass`] that owns its tokens, as [`Pass::rows`] hands them back.
#[derive(Debug, Clone)]
pub struct OwnedPass {
    pub ids: Vec<u32>,
    pub positions: Vec<u32>,
    pub segments: Vec<u32>,
    pub readout: Vec<usize>,
}

impl OwnedPass {
    /// Borrow it as a [`Pass`], to run it like any other.
    pub fn as_pass(&self) -> Pass<'_> {
        Pass {
            ids: &self.ids,
            positions: &self.positions,
            segments: &self.segments,
            readout: &self.readout,
        }
    }
}

/// Everything the engine needs from an inference backend.
///
/// Deliberately narrow, and deliberately not a generation API: Kev generates
/// nothing. One forward pass over a backbone (the base model with the
/// checkpoint's LoRA adapter applied, no vocabulary head), and the hidden states
/// at the positions the readout asks for.
pub trait Forward: Send {
    /// Token ids for a piece of caller text.
    ///
    /// Must never return a special token: the text is escaped before it gets
    /// here, so a tokenizer that treats `<|...|>` as ordinary text is enough.
    fn tokenise(&mut self, text: &str) -> Result<Vec<u32>>;

    /// The token id of one of Kev's five delimiters ([`SPECIAL`]).
    fn delimiter(&mut self, token: &str) -> Result<u32>;

    /// Hidden states at `pass.readout`, in that order, one vector of the
    /// backbone's hidden size each.
    ///
    /// The mask is [`Pass::attends`], the position ids are `pass.positions`, and
    /// the last layer's output is what the pointer head reads — not logits over
    /// the vocabulary. A backend that cannot honour the mask runs
    /// [`Pass::rows`].
    fn hidden(&mut self, pass: &Pass<'_>) -> Result<Vec<Vec<f32>>>;
}

/// Kev run in-process, over some [`Forward`] backend.
///
/// `Clone` is load-bearing and cheap: the [`SystemOne`] impl hands an owned
/// copy to `spawn_blocking`, which needs `'static`. The backend itself is
/// shared, not copied — one model in memory, however many handles.
#[derive(Clone)]
pub struct LocalEngine {
    backend: Arc<Mutex<dyn Forward>>,
    head: Arc<PointerHead>,
    model: String,
    limits: Limits,
}

impl LocalEngine {
    /// An engine over an inference backend and a checkpoint's pointer head.
    pub fn new(backend: impl Forward + 'static, head: PointerHead) -> Self {
        Self {
            backend: Arc::new(Mutex::new(backend)),
            head: Arc::new(head),
            model: String::from(crate::DEFAULT_MODEL),
            limits: Limits::serving(),
        }
    }

    /// The name reported in responses that do not pin a model themselves.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Shorten the context. The default is [`Limits::serving`]; the released
    /// checkpoints were trained on [`Limits::training`].
    pub fn with_limits(mut self, limits: Limits) -> Self {
        self.limits = limits;
        self
    }

    /// Answer a request on the calling thread.
    ///
    /// Deliberately blocking: a forward pass is CPU/GPU-bound and has no
    /// business on an async runtime thread. The [`SystemOne`] impl wraps this
    /// in `spawn_blocking`; callers on a runtime other than tokio can call it
    /// from their own blocking context.
    pub fn system_one_blocking(&self, request: &SystemOneRequest) -> Result<SystemOneResponse> {
        let started = Instant::now();
        let (record, plans) = prompt::plan(request);
        let mut backend = self.locked()?;

        let encoding = self.encode(&mut *backend, &record)?;
        let probabilities = self.probabilities(&mut *backend, &encoding)?;
        let answers = self.answers(&probabilities, &plans)?;

        self.respond(request, &mut *backend, answers, encoding.ids.len(), started)
    }

    /// Answer a request with one forward pass per question.
    ///
    /// Each question is asked on its own, against the same state, the way
    /// `/v1/systemone/separate` does it: the state is encoded once per question,
    /// so `usage.input_tokens` counts it once per question too. Questions are
    /// already isolated in the packed call — this exists to verify that.
    pub fn system_one_separate_blocking(
        &self,
        request: &SystemOneRequest,
    ) -> Result<SystemOneResponse> {
        let started = Instant::now();
        let (record, plans) = prompt::plan(request);
        let mut backend = self.locked()?;

        let mut answers = IndexMap::with_capacity(plans.len());
        let mut input_tokens = 0;
        for (question, plan) in record.questions.iter().zip(&plans) {
            let alone = Record {
                state: record.state.clone(),
                questions: vec![question.clone()],
            };
            let encoding = self.encode(&mut *backend, &alone)?;
            let probabilities = self.probabilities(&mut *backend, &encoding)?;
            input_tokens += encoding.ids.len();
            answers.insert(plan.id.clone(), readout::answer(&probabilities[0], plan)?);
        }

        self.respond(request, &mut *backend, answers, input_tokens, started)
    }

    fn encode(&self, backend: &mut dyn Forward, record: &Record) -> Result<Encoding> {
        let delimiters = Delimiters {
            state: backend.delimiter(SPECIAL[0])?,
            question: backend.delimiter(SPECIAL[1])?,
            option: backend.delimiter(SPECIAL[2])?,
            option_end: backend.delimiter(SPECIAL[3])?,
            decide: backend.delimiter(SPECIAL[4])?,
        };
        encode::encode(record, &delimiters, self.limits, |text| {
            backend.tokenise(text)
        })
    }

    /// One forward pass, then the pointer head: a distribution per question.
    fn probabilities(
        &self,
        backend: &mut dyn Forward,
        encoding: &Encoding,
    ) -> Result<Vec<Vec<f32>>> {
        let readout = encoding.readout();
        let pass = Pass {
            ids: &encoding.ids,
            positions: &encoding.positions,
            segments: &encoding.segments,
            readout: &readout,
        };
        let hidden = backend.hidden(&pass)?;
        if hidden.len() != readout.len() {
            return Err(Error::Engine(format!(
                "the backend returned {} hidden states for {} readout positions",
                hidden.len(),
                readout.len()
            )));
        }

        let mut hidden = hidden.into_iter();
        (0..encoding.decide.len())
            .map(|question| {
                // The readout is laid out per question: <decide> first, then
                // one `</opt>` per option.
                let mut states = hidden.by_ref().take(encoding.readout_width(question));
                let decide = states.next().expect("readout width is at least one");
                let options: Vec<Vec<f32>> = states.collect();
                Ok(readout::softmax(&self.head.logits(&decide, &options)?))
            })
            .collect()
    }

    fn answers(
        &self,
        probabilities: &[Vec<f32>],
        plans: &[Plan],
    ) -> Result<IndexMap<String, Answer>> {
        probabilities
            .iter()
            .zip(plans)
            .map(|(p, plan)| Ok((plan.id.clone(), readout::answer(p, plan)?)))
            .collect()
    }

    fn respond(
        &self,
        request: &SystemOneRequest,
        backend: &mut dyn Forward,
        answers: IndexMap<String, Answer>,
        input_tokens: usize,
        started: Instant,
    ) -> Result<SystemOneResponse> {
        // Not generated tokens - there are none. The server bills the
        // serialised answers, so this counts the same string.
        let output_tokens = backend.tokenise(&readout::answers_json(&answers))?.len();
        Ok(SystemOneResponse {
            model: request.model.clone().unwrap_or_else(|| self.model.clone()),
            answers,
            usage: Usage {
                input_tokens: input_tokens as u64,
                output_tokens: output_tokens as u64,
            },
            latency_ms: Some((started.elapsed().as_secs_f64() * 10_000.0).round() / 10.0),
            // A header the HTTP server sets; there is no server here.
            request_id: None,
        })
    }

    /// A panic in one forward pass must not wedge every later call, so say so
    /// plainly instead of propagating the poison.
    ///
    /// The `+ 'static` is not decoration: the field holds `dyn Forward + 'static`,
    /// a bare `dyn Forward` here would elide to the lifetime of `&self`, and
    /// `MutexGuard` is invariant over `T`, so the two do not convert.
    fn locked(&self) -> Result<std::sync::MutexGuard<'_, dyn Forward + 'static>> {
        self.backend.lock().map_err(|_| {
            Error::Engine(String::from(
                "the inference backend is poisoned: an earlier forward pass panicked",
            ))
        })
    }
}

impl fmt::Debug for LocalEngine {
    // The backend is an opaque model handle; there is nothing useful to print.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LocalEngine")
            .field("model", &self.model)
            .finish_non_exhaustive()
    }
}

// Kept as `-> impl Future + Send` rather than `async fn`: it mirrors the trait
// signature exactly, so the Send bound stays visible at every implementation
// site instead of being something the compiler infers out of sight.
#[allow(clippy::manual_async_fn)]
impl SystemOne for LocalEngine {
    fn system_one(
        &self,
        request: &SystemOneRequest,
    ) -> impl Future<Output = Result<SystemOneResponse>> + Send {
        let engine = self.clone();
        let request = request.clone();
        async move { blocking(move || engine.system_one_blocking(&request)).await }
    }

    fn system_one_separate(
        &self,
        request: &SystemOneRequest,
    ) -> impl Future<Output = Result<SystemOneResponse>> + Send {
        let engine = self.clone();
        let request = request.clone();
        async move { blocking(move || engine.system_one_separate_blocking(&request)).await }
    }
}

/// Run blocking work off the runtime thread, turning a panic into an error
/// rather than taking the caller's task down with it.
async fn blocking<F>(work: F) -> Result<SystemOneResponse>
where
    F: FnOnce() -> Result<SystemOneResponse> + Send + 'static,
{
    tokio::task::spawn_blocking(work)
        .await
        .map_err(|e| Error::Engine(format!("the forward pass did not finish: {e}")))?
}
