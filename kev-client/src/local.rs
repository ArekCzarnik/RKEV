//! A local inference engine: the model runs in this process, no HTTP.
//!
//! **Skeleton.** [`LocalEngine`] knows the shape of the work — blocking on the
//! inside, the [`SystemOne`] seam on the outside — but not yet how to do it.
//! The prompt build and the logit readout are still missing; see
//! `.claude/tasks/2026-09-23-local-inference.md`.
//!
//! The inference backend sits behind [`Forward`]. mistral.rs is the chosen
//! engine and will implement that trait; keeping it to one impl means the
//! choice can still be revisited without touching anything else.

use std::fmt;
use std::future::Future;
use std::sync::{Arc, Mutex};

use crate::error::{Error, Result};
use crate::system_one::SystemOne;
use crate::types::{SystemOneRequest, SystemOneResponse};

/// Everything the engine needs from an inference backend.
///
/// Deliberately narrow: tokenise, and run one forward pass. No generation —
/// Kev generates nothing, it reads probabilities off the logits.
pub trait Forward: Send {
    /// Token ids for a piece of text.
    ///
    /// Needed twice over: to build the prompt, and to look up the candidate
    /// tokens a readout compares (the yes/no pair, one per choice option, the
    /// level indices).
    fn tokenise(&mut self, text: &str) -> Result<Vec<u32>>;

    /// Logits at each requested position, in the order asked for.
    ///
    /// `positions` index into `tokens`. A backend that can only report the
    /// final position can still serve a single-position request; whether Kev
    /// needs more than one is the open question that decides this signature.
    fn logits(&mut self, tokens: &[u32], positions: &[usize]) -> Result<Vec<Vec<f32>>>;
}

/// Kev run in-process, over some [`Forward`] backend.
///
/// `Clone` is load-bearing and cheap: the [`SystemOne`] impl hands an owned
/// copy to `spawn_blocking`, which needs `'static`. The backend itself is
/// shared, not copied — one model in memory, however many handles.
#[derive(Clone)]
pub struct LocalEngine {
    backend: Arc<Mutex<dyn Forward>>,
}

impl LocalEngine {
    /// An engine over an inference backend.
    pub fn new(backend: impl Forward + 'static) -> Self {
        Self {
            backend: Arc::new(Mutex::new(backend)),
        }
    }

    /// Answer a request on the calling thread.
    ///
    /// Deliberately blocking: a forward pass is CPU/GPU-bound and has no
    /// business on an async runtime thread. The [`SystemOne`] impl wraps this
    /// in `spawn_blocking`; callers on a runtime other than tokio can call it
    /// from their own blocking context.
    pub fn system_one_blocking(&self, _request: &SystemOneRequest) -> Result<SystemOneResponse> {
        let mut backend = self.locked()?;

        // TODO(PR 4-7): serialise state + questions into a prompt, tokenise it
        // here, one `logits` call, then read a distribution per question type.
        // Blocked on the prompt template and the readout - see the task file.
        // An error rather than a panic, so callers can be written and tested
        // against the seam today.
        let _ = backend.tokenise("");

        Err(Error::Engine(String::from(
            "the local engine has no forward pass yet: the prompt template and \
             logit readout still have to be taken from the Python kev",
        )))
    }

    /// Answer a request with one forward pass per question.
    pub fn system_one_separate_blocking(
        &self,
        request: &SystemOneRequest,
    ) -> Result<SystemOneResponse> {
        // TODO(PR 8): one pass per question. Same blocker.
        self.system_one_blocking(request)
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
        f.debug_struct("LocalEngine").finish_non_exhaustive()
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
