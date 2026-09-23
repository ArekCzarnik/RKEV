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
//! use kev_client::{Choice, Client, Noul, Score, SystemOneRequest};
//!
//! # async fn run() -> Result<(), kev_client::Error> {
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

mod client;
mod error;
mod types;

pub use client::{Client, DEFAULT_BASE_URL, DEFAULT_MODEL};
pub use error::{Error, Result};
pub use types::{
    Answer, Choice, Noul, NoulCriteria, Question, Score, SystemOneRequest, SystemOneResponse, Usage,
};

pub use indexmap::IndexMap;
