//! A Rust client for [Kev](https://github.com/jaredpalmer/kev), the small
//! decision models that speak TypeSafe's System One API.
//!
//! You send a `state` — a support ticket, a document, any text — plus a set of
//! questions about it, and get back probabilities instead of a single label.
//! Questions share the state but cannot read each other.
//!
//! Start a server first:
//!
//! ```text
//! uv run --extra serve python -m kev.serve --run jaredpalmer/kev-4b --port 8009
//! ```
//!
//! Then:
//!
//! ```no_run
//! # #[cfg(feature = "http")]
//! # async fn run() -> Result<(), kev_client::Error> {
//! use kev_client::{Choice, Client, Noul, Score, SystemOneRequest};
//!
//! let client = Client::local()?;
//!
//! let response = client
//!     .system_one(
//!         &SystemOneRequest::new(
//!             "Shoes arrived two weeks late and in the wrong size. \
//!              Also I see two charges on my card.",
//!         )
//!         .ask(
//!             "department",
//!             Choice::new("Which team should handle this?")
//!                 .option("returns", "Exchanges, refunds, wrong or damaged items")
//!                 .option("shipping", "Delivery status, delays, lost packages")
//!                 .option("billing", "Charges, invoices, payment problems"),
//!         )
//!         .ask("escalate", Noul::new("Does this need urgent human attention?"))
//!         .ask(
//!             "frustration",
//!             Score::new("How frustrated is the customer?")
//!                 .level("Calm")
//!                 .level("Frustrated")
//!                 .level("Very angry"),
//!         ),
//!     )
//!     .await?;
//!
//! if let Some(answer) = response.answer("department") {
//!     println!("{:?} (confidence {:?})", answer.as_choice(), answer.confidence());
//! }
//! # Ok(())
//! # }
//! ```

#[cfg(feature = "qwen3")]
mod backend;
#[cfg(feature = "http")]
mod client;
#[cfg(feature = "local")]
mod encode;
mod error;
#[cfg(feature = "local")]
mod local;
#[cfg(feature = "local")]
mod prompt;
#[cfg(feature = "qwen3")]
mod qwen3;
#[cfg(feature = "local")]
mod readout;
mod system_one;
mod types;

/// Model alias a Kev server resolves to whatever checkpoint it loaded, and the
/// name a local engine reports when a request does not pin one.
pub const DEFAULT_MODEL: &str = "kev-latest";

#[cfg(feature = "qwen3")]
pub use backend::{pointer_head, Qwen3Backend};
#[cfg(feature = "http")]
pub use client::{Client, DEFAULT_BASE_URL};
#[cfg(feature = "local")]
pub use encode::{Limits, DECIDE, OPTION, OPTION_END, QUESTION, SPECIAL, STATE};
pub use error::{Error, Result};
#[cfg(feature = "local")]
pub use local::{Forward, LocalEngine, OwnedPass, Pass};
#[cfg(feature = "local")]
pub use prompt::render;
#[cfg(feature = "local")]
#[cfg(feature = "qwen3")]
pub use qwen3::{Backbone, Config};
#[cfg(feature = "local")]
pub use readout::{softmax, Linear, PointerHead};
pub use system_one::SystemOne;
pub use types::{
    Answer, Choice, Noul, NoulCriteria, Question, Score, SystemOneRequest, SystemOneResponse, Usage,
};

pub use indexmap::IndexMap;
