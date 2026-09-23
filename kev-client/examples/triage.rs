//! Triage one support ticket, the worked example from the Kev README.
//!
//! Start a server first:
//!
//! ```text
//! uv run --extra serve python -m kev.serve --run jaredpalmer/kev-4b --port 8009
//! ```
//!
//! Then:
//!
//! ```text
//! cargo run --example triage
//! cargo run --example triage -- "My parcel never arrived and nobody answers."
//! ```
//!
//! `KEV_BASE_URL`, `KEV_MODEL` and `KEV_API_KEY` override the defaults.

use std::env;

use kev_client::{Answer, Choice, Client, Noul, Score, SystemOneRequest};

const DEFAULT_TICKET: &str = "Shoes arrived two weeks late and in the wrong size. \
                              Also I see two charges on my card.";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ticket = env::args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_TICKET.to_string());

    let mut client = Client::new(
        env::var("KEV_BASE_URL").unwrap_or_else(|_| kev_client::DEFAULT_BASE_URL.to_string()),
    )?;
    if let Ok(model) = env::var("KEV_MODEL") {
        client = client.with_model(model);
    }
    if let Ok(key) = env::var("KEV_API_KEY") {
        client = client.with_api_key(key);
    }

    let request = SystemOneRequest::new(ticket.as_str())
        .ask(
            "department",
            Choice::new("Which team should handle this?")
                .option("returns", "Exchanges, refunds, wrong or damaged items")
                .option("shipping", "Delivery status, delays, lost packages")
                .option("billing", "Charges, invoices, payment problems"),
        )
        .ask(
            "escalate",
            Noul::new("Does this need urgent human attention?"),
        )
        .ask(
            "frustration",
            Score::new("How frustrated is the customer?")
                .level("Calm")
                .level("Frustrated")
                .level("Very angry"),
        );

    let response = match client.system_one(&request).await {
        Ok(response) => response,
        Err(error) => {
            eprintln!("{error}");
            eprintln!(
                "\nIs a kev server running? Start one with:\n  \
                 uv run --extra serve python -m kev.serve --run jaredpalmer/kev-4b --port 8009"
            );
            std::process::exit(1);
        }
    };

    println!("ticket: {ticket}\n");

    for (id, answer) in &response.answers {
        match answer {
            Answer::Noul { noul } => {
                println!("{id:<12} yes with probability {noul:.2}");
            }
            Answer::Choice {
                choice, confidence, ..
            } => {
                println!("{id:<12} {choice} (confidence {confidence:.2})");
            }
            Answer::Score {
                score, confidence, ..
            } => {
                let label = answer
                    .legend()
                    .and_then(|legend| legend.get(&score.round().to_string()))
                    .map(String::as_str)
                    .unwrap_or("?");
                println!("{id:<12} {score:.2} ~ {label} (confidence {confidence:.2})");
            }
        }

        if let Some(probabilities) = answer.probabilities() {
            let breakdown: Vec<String> = probabilities
                .iter()
                .map(|(name, p)| format!("{name} {p:.2}"))
                .collect();
            println!("{:<12} {}", "", breakdown.join("  "));
        }
    }

    println!(
        "\n{} in / {} out tokens{}",
        response.usage.input_tokens,
        response.usage.output_tokens,
        response
            .latency_ms
            .map(|ms| format!(", {ms:.0} ms"))
            .unwrap_or_default()
    );

    Ok(())
}
