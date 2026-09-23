# kev-client

A Rust client for [Kev](https://github.com/jaredpalmer/kev) — small decision
models you run yourself. Kev implements TypeSafe's
[System One](https://docs.typesafe.ai/api) API, so this client works against
either.

You send a **state** (a ticket, a document, any text) plus a set of
**questions**, and get back probabilities rather than a single label. The
questions share the state but cannot read each other.

## Start a server

Kev itself is Python. From a checkout of the Kev repo:

```bash
uv sync --extra serve
uv run --extra serve python -m kev.serve --run jaredpalmer/kev-4b --port 8009
```

The first run downloads the adapter and the base model.

## Use it

```toml
[dependencies]
kev-client = { path = "../kev-client" }
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

```rust
use kev_client::{Choice, Client, Noul, Score, SystemOneRequest};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::local()?;

    let response = client
        .system_one(
            &SystemOneRequest::new(
                "Shoes arrived two weeks late and in the wrong size. \
                 Also I see two charges on my card.",
            )
            .ask(
                "department",
                Choice::new("Which team should handle this?")
                    .option("returns", "Exchanges, refunds, wrong or damaged items")
                    .option("shipping", "Delivery status, delays, lost packages")
                    .option("billing", "Charges, invoices, payment problems"),
            )
            .ask("escalate", Noul::new("Does this need urgent human attention?"))
            .ask(
                "frustration",
                Score::new("How frustrated is the customer?")
                    .level("Calm")
                    .level("Frustrated")
                    .level("Very angry"),
            ),
        )
        .await?;

    let department = response.answer("department").unwrap();
    println!("{:?} {:?}", department.as_choice(), department.top());

    Ok(())
}
```

## Question types

| Builder | Wire type | Answer |
|---|---|---|
| `Noul::new(instructions)` + `.yes(..)` / `.no(..)` | `noul` | `as_noul()` — probability of yes |
| `Choice::new(instructions)` + `.option(name, description)` / `.option_bare(name)` | `choice` | `as_choice()`, plus `confidence()` and `probabilities()` |
| `Score::new(instructions)` + `.level(description)` | `score` | `as_score()` — mean level index from 0 — plus `legend()`, `confidence()` and `probabilities()` |

`Answer::top()` gives the most likely label of the distribution, for `choice`
and `score` alike. `confidence()` is a shape measure of the distribution, not a
measured accuracy rate, and a `noul` answer has none — the probability itself is
the answer.

Option order is part of the request and can change the answer, so options and
questions keep the order you add them in.

## Run the example

```bash
cargo run --example triage
cargo run --example triage -- "My parcel never arrived and nobody answers."
```

`KEV_BASE_URL`, `KEV_MODEL` and `KEV_API_KEY` override the defaults
(`http://127.0.0.1:8009`, `kev-latest`, no auth). Set `KEV_API_KEY` when the
server runs with one — Kev then requires `Authorization: Bearer <key>` on
`/v1/*`.

## Endpoints covered

| Method | Path | Client |
|---|---|---|
| `POST` | `/v1/systemone` | `system_one` |
| `POST` | `/v1/systemone/separate` | `system_one_separate` |
| `POST` | `/v1/systemone/permute` | `permute` — raw JSON, the shape is not documented |
| `GET` | `/v1/models` | `models` — raw JSON, same reason |

## Without a server (feature `local`)

`LocalEngine` answers the same requests in this process: it builds Kev's prompt,
runs one forward pass and reads the answers off the pointer head. What it still
needs is an inference backend, which this crate does not ship:

```rust
use kev_client::{Forward, Pass, Result};

impl Forward for MyBackbone {
    // Token ids for caller text, and the ids of Kev's five delimiters.
    fn tokenise(&mut self, text: &str) -> Result<Vec<u32>> { .. }
    fn delimiter(&mut self, token: &str) -> Result<u32> { .. }

    // The backbone's last hidden states at `pass.readout`.
    fn hidden(&mut self, pass: &Pass<'_>) -> Result<Vec<Vec<f32>>> { .. }
}
```

That is the backbone — the base model with the checkpoint's LoRA adapter and no
vocabulary head — run with `pass.attends` as the attention mask and
`pass.positions` as the position ids. Kev generates nothing and its answers do
not come from logits: a checkpoint's output layer is a small pointer head that
scores each option's `</opt>` hidden state against its question's `<decide>`.
That head is the second half of a checkpoint, and it goes in alongside:

```rust
let head = PointerHead::new(query, key)?.with_temperature(2.3)?;
let engine = LocalEngine::new(backbone, head);
let response = engine.system_one_blocking(&request)?;
```

A local-only build carries no HTTP stack:

```toml
kev-client = { path = "../kev-client", default-features = false, features = ["local"] }
```

## Tests

```bash
cargo test
```

The tests check the request and response shapes against the worked example in
the Kev README; they need no server.
