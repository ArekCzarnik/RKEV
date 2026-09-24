//! Local inference for [Kev](https://github.com/jaredpalmer/kev), the small
//! decision models that speak TypeSafe's System One API.
//!
//! You hand it a `state` — a support ticket, a document, any text — plus a set of
//! questions about it, and get back probabilities instead of a single label.
//! Questions share the state but cannot read each other.
//!
//! Everything happens in this process: the prompt, one forward pass over a Qwen
//! backbone with the checkpoint's LoRA merged in, and the pointer head that turns
//! the hidden states into answers.
//!
//! ```no_run
//! # #[cfg(feature = "candle")]
//! # fn run() -> Result<(), rkev::Error> {
//! use std::path::Path;
//!
//! use rkev::{pointer_head, Backend, Choice, LocalEngine, Noul, Score, SystemOneRequest};
//!
//! let base = Path::new("models/qwen3-0.6b-base");
//! let checkpoint = Path::new("models/kev-0.6b");
//!
//! // The base model with the checkpoint's adapter, and the checkpoint's own
//! // pointer head: the two halves of a Kev model.
//! let engine = LocalEngine::new(
//!     Backend::open(base, Some(checkpoint))?,
//!     pointer_head(&checkpoint.join("head.pt"))?,
//! );
//!
//! let response = engine.system_one_blocking(
//!     &SystemOneRequest::new(
//!         "Shoes arrived two weeks late and in the wrong size. \
//!          Also I see two charges on my card.",
//!     )
//!     .ask(
//!         "department",
//!         Choice::new("Which team should handle this?")
//!             .option("returns", "Exchanges, refunds, wrong or damaged items")
//!             .option("shipping", "Delivery status, delays, lost packages")
//!             .option("billing", "Charges, invoices, payment problems"),
//!     )
//!     .ask("escalate", Noul::new("Does this need urgent human attention?"))
//!     .ask(
//!         "frustration",
//!         Score::new("How frustrated is the customer?")
//!             .level("Calm")
//!             .level("Frustrated")
//!             .level("Very angry"),
//!     ),
//! )?;
//!
//! if let Some(answer) = response.answer("department") {
//!     println!("{:?} (confidence {:?})", answer.as_choice(), answer.confidence());
//! }
//! # Ok(())
//! # }
//! ```
//!
//! [`LocalEngine::system_one`] is the same call for an async caller: it moves the
//! pass off the runtime thread, since a forward pass is CPU-bound.

#[cfg(feature = "candle")]
mod backend;
#[cfg(feature = "local")]
mod encode;
mod error;
#[cfg(feature = "local")]
mod local;
#[cfg(feature = "local")]
mod prompt;
/// The attention-only Qwen3 backbone (the `@qwen3` checkpoints).
#[cfg(feature = "candle")]
pub mod qwen3;
/// The hybrid Qwen3.5 backbone: attention mixed with Gated DeltaNet.
#[cfg(feature = "candle")]
pub mod qwen3_5;
#[cfg(feature = "local")]
mod readout;
mod system_one;
mod types;
#[cfg(feature = "candle")]
mod weights;

/// The name a local engine reports when a request does not pin one.
pub const DEFAULT_MODEL: &str = "kev-latest";

#[cfg(feature = "candle")]
pub use backend::{option_isolation, pointer_head, temperature, Backend};
#[cfg(feature = "local")]
pub use encode::{Limits, OptionSlot, DECIDE, OPTION, OPTION_END, QUESTION, SPECIAL, STATE};
pub use error::{Error, Result};
#[cfg(feature = "local")]
pub use local::{Forward, LocalEngine, OwnedPass, Pass};
#[cfg(feature = "local")]
pub use prompt::render;
#[cfg(feature = "local")]
pub use readout::{answers_json, softmax, Linear, PointerHead};
pub use system_one::SystemOne;
pub use types::{
    Answer, Choice, Noul, NoulCriteria, Question, Score, SystemOneRequest, SystemOneResponse, Usage,
};

pub use indexmap::IndexMap;
