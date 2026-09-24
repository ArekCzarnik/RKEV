//! The backend seam.
//!
//! What every backend has in common, so callers can be written once and keep
//! working when the backend underneath changes.

use std::future::Future;

use crate::error::Result;
use crate::types::{SystemOneRequest, SystemOneResponse};

/// The System One API, independent of how it is served.
///
/// [`LocalEngine`](crate::LocalEngine) implements it (feature `local`). Anything
/// else that can answer these two calls — another engine, a queue in front of
/// one, a recording — fits the same seam.
///
/// The methods return `impl Future + Send` instead of being `async fn`: Rust
/// 1.75 cannot put a `Send` bound on an `async fn` in a trait, and without it
/// callers cannot `tokio::spawn` a request. The cost is that the trait is not
/// dyn-compatible — there is no `Box<dyn SystemOne>`. Generic code covers most
/// uses; a backend picked at runtime wants an enum over the backends instead.
pub trait SystemOne {
    /// Ask all questions in one forward pass.
    fn system_one(
        &self,
        request: &SystemOneRequest,
    ) -> impl Future<Output = Result<SystemOneResponse>> + Send;

    /// Ask each question in its own forward pass.
    ///
    /// Questions are already isolated from each other in the normal call; this
    /// exists to verify that, and costs one pass per question.
    fn system_one_separate(
        &self,
        request: &SystemOneRequest,
    ) -> impl Future<Output = Result<SystemOneResponse>> + Send;
}
