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

use crate::encode::{self, Delimiters, Encoding, Limits, OptionSlot, SPECIAL};
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
    /// Which option span each token belongs to, for a checkpoint trained with
    /// option isolation. Empty otherwise, which is the usual case.
    pub options: &'a [OptionSlot],
}

impl<'a> Pass<'a> {
    /// A pass with no option isolation, which is what a checkpoint normally
    /// wants.
    pub fn new(
        ids: &'a [u32],
        positions: &'a [u32],
        segments: &'a [u32],
        readout: &'a [usize],
    ) -> Self {
        Self {
            ids,
            positions,
            segments,
            readout,
            options: &[],
        }
    }

    /// The same pass, with each token's option span named — the layout an
    /// option-isolated checkpoint was trained on.
    pub fn with_options(mut self, options: &'a [OptionSlot]) -> Self {
        self.options = options;
        self
    }

    /// Whether this pass carries option isolation.
    pub fn is_isolated(&self) -> bool {
        !self.options.is_empty()
    }
}

impl Pass<'_> {
    /// Whether the token at `query` may read the token at `key`.
    ///
    /// Causal *and* blind across questions. Questions sharing one pass must not
    /// see each other, so this is not a plain causal mask: a backend that
    /// cannot express it must run [`Pass::rows`] instead.
    ///
    /// Under option isolation there is one more rule: an option's tokens are
    /// read by that option and by `<decide>`, and by nothing else. That is what
    /// keeps one option from conditioning another.
    pub fn attends(&self, query: usize, key: usize) -> bool {
        if key > query || (self.segments[key] != 0 && self.segments[key] != self.segments[query]) {
            return false;
        }
        match self.options.get(key) {
            Some(OptionSlot::Option(span)) => {
                query == key
                    || matches!(self.options[query], OptionSlot::Decide)
                    || self.options[query] == OptionSlot::Option(*span)
            }
            _ => true,
        }
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
            // The state holds no options, isolated or not.
            options: Vec::new(),
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
                options: indices
                    .iter()
                    .filter_map(|i| self.options.get(*i).copied())
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
                options: indices
                    .iter()
                    .filter_map(|i| self.options.get(*i).copied())
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
    pub options: Vec<OptionSlot>,
}

impl OwnedPass {
    /// Borrow it as a [`Pass`], to run it like any other.
    pub fn as_pass(&self) -> Pass<'_> {
        Pass::new(&self.ids, &self.positions, &self.segments, &self.readout)
            .with_options(&self.options)
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

    /// Hidden states for several passes at once.
    ///
    /// The default answers them one at a time. A backend overrides it when it
    /// can share work between them: several requests each have a state of their
    /// own, and prefilling them together is one pass instead of one each.
    fn hidden_batch(&mut self, passes: &[Pass<'_>]) -> Result<Vec<Vec<Vec<f32>>>> {
        passes.iter().map(|pass| self.hidden(pass)).collect()
    }

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
    isolation: bool,
}

impl LocalEngine {
    /// An engine over an inference backend and a checkpoint's pointer head.
    pub fn new(backend: impl Forward + 'static, head: PointerHead) -> Self {
        Self {
            backend: Arc::new(Mutex::new(backend)),
            head: Arc::new(head),
            model: String::from(crate::DEFAULT_MODEL),
            limits: Limits::serving(),
            isolation: false,
        }
    }

    /// The name reported in responses that do not pin a model themselves.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Lay every option out as a sub-branch of its own, which is what a
    /// checkpoint trained with option isolation expects.
    ///
    /// Off by default, because the released checkpoints were trained without it —
    /// `head.pt` says which ([`option_isolation`](crate::option_isolation)), and
    /// getting it wrong is a silently different prompt rather than an error. It
    /// needs the packed mask, so a recurrent base refuses it.
    pub fn with_option_isolation(mut self, isolation: bool) -> Self {
        self.isolation = isolation;
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

    /// Answer several requests, prefilling their states together.
    ///
    /// Each request still gets its own answers; what is shared is the pass that
    /// runs the states. That is worth it when the states are short enough that a
    /// pass costs more in overhead than in arithmetic, and worth little when each
    /// state already fills the machine — measure before reaching for it.
    ///
    /// All or nothing: a request that does not fit the context fails the call,
    /// rather than leaving the caller to work out which answers belong to whom.
    pub fn system_one_batch_blocking(
        &self,
        requests: &[SystemOneRequest],
    ) -> Result<Vec<SystemOneResponse>> {
        let started = Instant::now();
        let planned: Vec<(Record, Vec<Plan>)> = requests.iter().map(prompt::plan).collect();
        let mut backend = self.locked()?;

        let encodings = planned
            .iter()
            .map(|(record, _)| self.encode(&mut *backend, record))
            .collect::<Result<Vec<_>>>()?;
        let readouts: Vec<Vec<usize>> = encodings.iter().map(Encoding::readout).collect();
        let passes: Vec<Pass<'_>> = encodings
            .iter()
            .zip(&readouts)
            .map(|(encoding, readout)| {
                Pass::new(
                    &encoding.ids,
                    &encoding.positions,
                    &encoding.segments,
                    readout,
                )
                .with_options(&encoding.slots)
            })
            .collect();

        let hidden = backend.hidden_batch(&passes)?;
        if hidden.len() != requests.len() {
            return Err(Error::Engine(format!(
                "the backend answered {} of {} passes",
                hidden.len(),
                requests.len()
            )));
        }

        hidden
            .into_iter()
            .zip(encodings.iter().zip(planned.iter().zip(requests)))
            .map(|(hidden, (encoding, ((_, plans), request)))| {
                let probabilities = self.distributions(hidden, encoding)?;
                let answers = self.answers(&probabilities, plans)?;
                self.respond(request, &mut *backend, answers, encoding.ids.len(), started)
            })
            .collect()
    }

    /// Run one `choice` question under several option orders, to see whether the
    /// order moves the answer.
    ///
    /// The same thing `/v1/systemone/permute` does, in the same JSON shape the
    /// [`Client`](crate::Client) hands back for it — `runs` (each with its
    /// `order`, `probabilities`, `choice` and `latency_ms`), `argmax_stable` and
    /// the per-option `spread`. Raw JSON for the same reason the client's is: the
    /// envelope is not in the API docs, and a struct here would invent a
    /// contract.
    ///
    /// The first run keeps the order as given and the rest are shuffled from
    /// `seed`, so a repeat with the same seed sees the same orders. They will not
    /// be the *server's* orders for that seed — this does not reimplement
    /// CPython's shuffle — and they do not need to be: what the endpoint is for is
    /// how far the probabilities move, not which permutations were tried.
    ///
    /// Only the named question is asked, as on the server, and `rounds` is
    /// clamped to 1..=64 as [`Client::permute`](crate::Client::permute) clamps
    /// `n_perm`. Every run repeats the same state, so the second one onwards costs
    /// only its own branch.
    pub fn permute_blocking(
        &self,
        request: &SystemOneRequest,
        question: &str,
        rounds: u8,
        seed: u64,
    ) -> Result<serde_json::Value> {
        let Some(crate::Question::Choice(choice)) = request.questions.get(question) else {
            return Err(Error::Invalid(format!(
                "{question:?} is not a choice question of this request"
            )));
        };
        let names: Vec<String> = choice.criteria.keys().cloned().collect();

        let mut runs = Vec::new();
        for round in 0..rounds.clamp(1, 64) {
            let order = if round == 0 {
                names.clone()
            } else {
                shuffled(&names, seed, u64::from(round))
            };
            let mut reordered = crate::Choice {
                instructions: choice.instructions.clone(),
                criteria: IndexMap::with_capacity(order.len()),
            };
            for name in &order {
                reordered
                    .criteria
                    .insert(name.clone(), choice.criteria[name].clone());
            }
            let mut one = request.clone();
            one.questions =
                std::iter::once((question.to_string(), crate::Question::Choice(reordered)))
                    .collect();

            let response = self.system_one_blocking(&one)?;
            let answer = response.answer(question).ok_or_else(|| {
                Error::Engine(format!("{question:?} went missing from its own answer"))
            })?;
            runs.push(serde_json::json!({
                "order": order,
                "probabilities": answer.probabilities(),
                "choice": answer.as_choice(),
                "latency_ms": response.latency_ms,
            }));
        }

        // How far each option's probability travelled across the orders, and
        // whether the winner ever changed.
        let at = |run: &serde_json::Value, name: &str| -> f64 {
            run["probabilities"][name].as_f64().unwrap_or(f64::NAN)
        };
        let spread: serde_json::Map<String, serde_json::Value> = names
            .iter()
            .map(|name| {
                let values: Vec<f64> = runs.iter().map(|run| at(run, name)).collect();
                let high = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let low = values.iter().copied().fold(f64::INFINITY, f64::min);
                (name.clone(), serde_json::json!(high - low))
            })
            .collect();
        let winners: std::collections::BTreeSet<&str> = runs
            .iter()
            .filter_map(|run| run["choice"].as_str())
            .collect();

        Ok(serde_json::json!({
            "runs": runs,
            "argmax_stable": winners.len() == 1,
            "spread": spread,
        }))
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
        encode::encode(record, &delimiters, self.limits, self.isolation, |text| {
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
        let pass = Pass::new(
            &encoding.ids,
            &encoding.positions,
            &encoding.segments,
            &readout,
        )
        .with_options(&encoding.slots);
        let hidden = backend.hidden(&pass)?;
        self.distributions(hidden, encoding)
    }

    /// The pointer head, over the hidden states one pass came back with: a
    /// distribution per question.
    fn distributions(&self, hidden: Vec<Vec<f32>>, encoding: &Encoding) -> Result<Vec<Vec<f32>>> {
        let wanted = encoding.readout().len();
        if hidden.len() != wanted {
            return Err(Error::Engine(format!(
                "the backend returned {} hidden states for {wanted} readout positions",
                hidden.len(),
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

// Kept out of the `SystemOne` trait on purpose: the shape this returns is not in
// the API docs, so it is not something to make every backend promise.
#[allow(clippy::manual_async_fn)]
impl LocalEngine {
    /// [`LocalEngine::permute_blocking`], off the runtime thread.
    pub fn permute(
        &self,
        request: &SystemOneRequest,
        question: &str,
        rounds: u8,
        seed: u64,
    ) -> impl Future<Output = Result<serde_json::Value>> + Send {
        let engine = self.clone();
        let request = request.clone();
        let question = question.to_string();
        async move { blocking(move || engine.permute_blocking(&request, &question, rounds, seed)).await }
    }
}

/// One option order, shuffled from a seed.
///
/// A small deterministic generator and a Fisher-Yates pass: enough to try
/// different orders reproducibly, and not pretending to be CPython's shuffle.
fn shuffled(names: &[String], seed: u64, round: u64) -> Vec<String> {
    let mut state = seed
        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
        .wrapping_add(round.wrapping_add(1));
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut order = names.to_vec();
    for index in (1..order.len()).rev() {
        order.swap(index, (next() % (index as u64 + 1)) as usize);
    }
    order
}

/// Run blocking work off the runtime thread, turning a panic into an error
/// rather than taking the caller's task down with it.
async fn blocking<T, F>(work: F) -> Result<T>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T> + Send + 'static,
{
    tokio::task::spawn_blocking(work)
        .await
        .map_err(|e| Error::Engine(format!("the forward pass did not finish: {e}")))?
}
