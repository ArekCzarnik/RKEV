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

Kev can also be trained to lay each option out as a sub-branch of its own, so
that an option's representation cannot depend on which options came before it.
`head.pt` says whether a checkpoint was (`option_isolation()`); the released ones
were not, and the engine follows it only when told:

```rust
let engine = LocalEngine::new(backend, head).with_option_isolation(true);
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

| Method | Path | Client | Local engine |
|---|---|---|---|
| `POST` | `/v1/systemone` | `system_one` | `system_one` |
| `POST` | `/v1/systemone/separate` | `system_one_separate` | `system_one_separate` |
| `POST` | `/v1/systemone/permute` | `permute` — raw JSON, the shape is not documented | `permute`, the same shape |
| `GET` | `/v1/models` | `models` — raw JSON, same reason | — |

`permute` runs one `choice` question under several option orders and reports how
far each probability travelled (`spread`) and whether the winner ever changed
(`argmax_stable`) — the question of whether option order moves the answer, which
it can. The local engine answers it in the same JSON shape, so the two backends
swap; the first run keeps the order as given, the rest are shuffled from a seed,
and every run repeats the same state, so only the first pays for it.

## Without a server (features `local`, `candle`)

`LocalEngine` answers the same requests in this process: it builds Kev's prompt,
runs one forward pass and reads the answers off the pointer head.

With the `candle` feature it comes with the model: candle, CPU by default, for
both generations of Kev's bases. Which one a checkpoint needs is in its
`config.json`, so there is nothing to choose:

```rust
use kev_client::{pointer_head, LocalEngine, Backend};

let backend = Backend::open(base_model_dir, Some(checkpoint_dir))?;
// The other half of a checkpoint: head.pt's two projections, and the
// temperature it was calibrated with.
let head = pointer_head(&checkpoint_dir.join("head.pt"))?;

let response = LocalEngine::new(backend, head).system_one_blocking(&request)?;
```

There is a command-line front end for it too, which is the whole server's job
done in process — no HTTP, no Python:

```bash
cargo run --release --features candle --example decide -- \
    --base ~/models/qwen3-4b-base --checkpoint ~/models/kev-4b \
    --state "Shoes arrived two weeks late and in the wrong size. \
             Also I see two charges on my card."
```

which prints a line per question and the rest of each distribution under it:

```text
department     shipping     0.47   confidence 0.21
               returns      0.28
               billing      0.25
escalate       yes          0.93
frustration    Frustrated   level 1.44   confidence 0.78
```

The numbers there are the ones the Kev README publishes for Kev-4B, not a run of
this code: what has been run here is the shape.

`--request request.json` answers a System One request exactly as you would POST
it (repeat it for a batch); `--questions questions.json` asks your own questions
about a `--state`; `--lines` reads a state per line from stdin and answers each as
it arrives, so the model loads once and the state cache stays warm; `--json`
prints what the server would have replied, answers serialised the way the Python
serialises them. `--dtype`, `--separate` and `--permute` are there as well, and
the option layout comes from `head.pt` unless you override it. Diagnostics go to
stderr, answers to stdout.

The Qwen3 bases (`jaredpalmer/kev-4b@qwen3`, `kev-8b`, `kev-0.6b`) are attention
only, so a whole request runs as one masked pass. The current bases (Qwen3.5) mix
attention with Gated DeltaNet layers, which are recurrent: a recurrence carries
state forward token by token and cannot be told to skip another question's
tokens, so every question runs as its own row — the state, then its branch. That
is exact rather than masked, and it is what the Python does there too.

The state is run once per request and every question continues from it, and the
last few states are kept across requests, keyed by their tokens — a repeated
document then costs only its questions:

```rust
let backend = Backend::open(base, Some(checkpoint))?
    .with_prefix_cache(8)        // states kept across requests; 4 by default
    .with_prefix_min_tokens(0);  // prefill even short states
let (hits, misses) = backend.prefix_hits();
```

The questions' branches then run as one padded batch, and on a recurrent base the
delta rule runs in chunks of 64 tokens rather than token by token. Both are exact;
between them and the prefix, a five-question request over a 500-token state went
from about two seconds to under half of one here — on a model with the released
checkpoints' widths, but made-up weights, so take it as a ratio.

Several requests at once share the pass that runs their states:

```rust
let answers = engine.system_one_batch_blocking(&requests)?;
```

That is worth something while the states are short — eight 20-token states ran
about 1.4x faster batched here — and nothing when each state already fills the
machine, so measure before reaching for it.

Precision follows the device, as the Python server does it: bf16 on a GPU, f32 on
the CPU, which is the path every published number was measured at. The LoRA is
merged in f32 before the cast, and the delta rule, the gated norm and the pointer
head stay in f32 whatever the backbone runs in.

```rust
use candle_core::{DType, Device};

let backend = Backend::open_as(base, Some(checkpoint), Device::Cpu, DType::F16)?;
```

candle has no bf16 matmul on a CPU, so that combination is refused rather than
failing deep inside a projection; f16 is the reduced precision a CPU can run
(about 1.2x faster here, and within 0.01 of f32 on the answers). Nothing is
quantised, and no device other than the CPU has been exercised.

Both forward passes are checked against transcriptions of Hugging Face's
`modeling_qwen3.py` and `modeling_qwen3_5.py`, so the arithmetic is the
arithmetic transformers defines.
A published checkpoint has been loaded and answered with: `jaredpalmer/kev-0.6b`
over `Qwen/Qwen3-0.6B-Base` in f32 on an Apple CPU takes all seven of the
unambiguous tickets in `examples/sanity.rs`, with `shipping` and `returns` at
1.00 where they belong and `p(yes)` at 0.71 against 0.01 on a thank-you note, and
the packed and prefilled paths agree to five decimals over the real layer stack.
What has *not* happened is a comparison with the Python answering the same
request, so treat the probabilities as unverified against the reference until it
has. `examples/parity.rs` is that run: it answers a request in process and
compares every probability with a recorded server response.

```bash
KEV_DTYPE=fp32 uv run --extra serve python -m kev.serve --run jaredpalmer/kev-4b@qwen3 --port 8009
curl -s localhost:8009/v1/systemone -H 'content-type: application/json' -d @request.json > server.json

cargo run --features candle --example parity -- \
    --base <the base model directory> --checkpoint <the kev checkpoint> \
    --request request.json --server server.json
```

Without a server — and without Python — two things are still checkable, and
`examples/sanity.rs` runs both against a real checkpoint:

```bash
cargo run --features candle --example sanity -- \
    --base <the base model directory> --checkpoint <the kev checkpoint>
```

It answers the same request along different paths — every question at once
against one at a time, the state prefilled once against per question, and on a
hybrid base the delta rule in chunks against token by token — and the numbers
have to match. That compares the implementation with itself, but the paths walk
different code, so a mistake in the layout, the mask or the recurrence shows up
as a disagreement. Then it answers a handful of tickets whose answer is not in
doubt. A checkpoint that scores 0.87 on held-out data should get nearly all of
them; an implementation that loads the weights but gets the prompt, the rotary
convention or the readout wrong sits near chance with distributions flat to two
decimals, which is exactly what it looks like on untrained weights:

```text
My parcel never arrived and the tracking has not moved in a…   billing (0.33, confidence 0.00)  department = shipping MISSED
```

The exit code is non-zero if a path disagrees or more than one ticket is missed.
This is not parity: it cannot tell you that the answers match the server, only
that they are self-consistent and mean something.

One script runs everything that can be checked this way, downloads included:

```bash
scripts/local.sh                                  # the offline suite; no weights
scripts/local.sh --fetch jaredpalmer/kev-0.6b     # fetch one with curl, then all of it
scripts/local.sh --checkpoint ~/models/kev-0.6b --measure
```

It downloads with `curl` — the `hf` CLI is itself Python — taking the adapter, the
head and the tokenizer, then the base model that `adapter_config.json` names,
sharded weights included. Then: the offline suite, the sanity example, the suite
again with `KEV_TOKENIZER` set to the real vocabulary, one request through
`decide`, and with `--measure` the timings. `--help` lists the rest.

To put another engine underneath instead, implement `Forward` yourself:

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

That is a backbone — the base model with the checkpoint's LoRA adapter and no
vocabulary head — run with `pass.attends` as the attention mask and
`pass.positions` as the position ids. Kev generates nothing and its answers do
not come from logits: a checkpoint's output layer is a small pointer head that
scores each option's `</opt>` hidden state against its question's `<decide>`.
That head is the second half of a checkpoint, and it goes in alongside:

```rust
let head = PointerHead::new(query, key)?.with_temperature(2.3)?;
let engine = LocalEngine::new(backbone, head);
```

A local-only build carries no HTTP stack:

```toml
kev-client = { path = "../kev-client", default-features = false, features = ["candle"] }
```

## Tests

```bash
cargo test
```

The tests check the request and response shapes against the worked example in
the Kev README; they need no server. With the `candle` feature they also run the
backbone over a checkpoint they write themselves. Two checks want a real Qwen
tokenizer, which is not vendored here:

```bash
KEV_TOKENIZER=/path/to/tokenizer.json cargo test --features candle
```
