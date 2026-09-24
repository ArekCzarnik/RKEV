# rkev

[Kev](https://github.com/jaredpalmer/kev) in Rust — small decision models you run
yourself, in this process. Kev implements TypeSafe's
[System One](https://docs.typesafe.ai/api) API, and this crate answers the same
requests locally. Kev's own implementation is *the
reference* throughout this README: the rules here are copied from it rather than
invented, and it is needed for exactly one thing, which is checking that the
numbers match (see **Checking the engine against the server**).

You hand it a **state** (a ticket, a document, any text) plus a set of
**questions**, and get back probabilities rather than a single label. The
questions share the state but cannot read each other.

## Use it

```toml
[dependencies]
rkev = { path = "../rkev" }
```

```rust
use std::path::Path;

use rkev::{pointer_head, Backend, Choice, LocalEngine, Noul, Score, SystemOneRequest};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let base = Path::new("models/qwen3-0.6b-base");
    let checkpoint = Path::new("models/kev-0.6b");

    // The base model with the checkpoint's adapter, and the checkpoint's own
    // pointer head: the two halves of a Kev model.
    let engine = LocalEngine::new(
        Backend::open(base, Some(checkpoint))?,
        pointer_head(&checkpoint.join("head.pt"))?,
    );

    let response = engine.system_one_blocking(
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
    )?;

    let department = response.answer("department").unwrap();
    println!("{:?} {:?}", department.as_choice(), department.top());

    Ok(())
}
```

An async caller uses `engine.system_one(&request).await` instead, which moves the
pass off the runtime thread — a forward pass is CPU-bound and has no business on
one. `LocalEngine` is cheap to clone, and clones share the one loaded model.

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

## The examples

| Example | For | Needs |
|---|---|---|
| `decide` | answering requests — the server's job in this process | a checkpoint |
| `sanity` | the engine against itself, and against tickets whose answer is not in doubt | a checkpoint |
| `measure` | f16 against f32, the prefix cache, chunking, batching | a checkpoint |
| `parity` | every probability against a recorded server response | a recording |
| `deutsch` | the same thing in German, and with `--vergleich` the same content in English beside it — the checkpoints are published on English data and the base is multilingual, so what that costs is worth measuring rather than assuming | a checkpoint |
| `eval` | accuracy and calibration on your own labelled records | a checkpoint and a JSONL set |

## What it answers

| The reference's endpoint | Here |
|---|---|
| `POST /v1/systemone` | `system_one`, `system_one_blocking` |
| `POST /v1/systemone/separate` | `system_one_separate`, `system_one_separate_blocking` |
| `POST /v1/systemone/permute` | `permute`, `permute_blocking` — raw JSON, since that envelope is not documented |

`permute` runs one `choice` question under several option orders and reports how
far each probability travelled (`spread`) and whether the winner ever changed
(`argmax_stable`) — the question of whether option order moves the answer, which
it can. The first run keeps the order as given, the rest are shuffled from a seed,
and every run repeats the same state, so only the first pays for it.

Several requests at once are `system_one_batch_blocking`, which has no endpoint
behind it: it is what a server would call for a batch it collected itself.

## Inside the engine (features `local`, `candle`)

`LocalEngine` answers the same requests in this process: it builds Kev's prompt,
runs one forward pass and reads the answers off the pointer head.

With the `candle` feature it comes with the model: candle, CPU by default, for
both generations of Kev's bases. Which one a checkpoint needs is in its
`config.json`, so there is nothing to choose:

```rust
use rkev::{pointer_head, LocalEngine, Backend};

let backend = Backend::open(base_model_dir, Some(checkpoint_dir))?;
// The other half of a checkpoint: head.pt's two projections, and the
// temperature it was calibrated with.
let head = pointer_head(&checkpoint_dir.join("head.pt"))?;

let response = LocalEngine::new(backend, head).system_one_blocking(&request)?;
```

There is a command-line front end for it too, which is the whole server's job
done in process:

```bash
cargo run --release --example decide -- \
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
prints what the server would have replied, answers serialised the way the
reference serialises them. `--dtype`, `--separate` and `--permute` are there as well, and
the option layout comes from `head.pt` unless you override it. Diagnostics go to
stderr, answers to stdout.

The Qwen3 bases (`jaredpalmer/kev-4b@qwen3`, `kev-8b`, `kev-0.6b`) are attention
only, so a whole request runs as one masked pass. The current bases (Qwen3.5) mix
attention with Gated DeltaNet layers, which are recurrent: a recurrence carries
state forward token by token and cannot be told to skip another question's
tokens, so every question runs as its own row — the state, then its branch. That
is exact rather than masked, and it is what the reference does there too.

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

All of this is measurable on your own checkpoint rather than on the fixtures —
what the prefix, its cache and f16 are worth on the weights you will serve:

```bash
cargo run --release --example measure -- \
    --base <base> --checkpoint <kev> [--words 400] [--repeat 3] [--batch 4]
```

Every row is a median over `--repeat` passes, each cold pass over its own state so
it is a real cache miss. Two of the rows are controls: the same configuration with
nothing kept between requests, so a cached row that is not far below it is not
measuring the cache, and f16's largest difference from f32 on the same answers, so
a speed-up that costs the probabilities is visible in the same table. The prefix
threshold is forced to zero, since an attention-only base otherwise skips the
prefix below 384 state tokens and both prefix rows would quietly measure the
packed path.

Precision follows the device, as the reference does it: bf16 on a GPU, f32 on
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
What has *not* happened is a comparison with the reference answering the same
request, so treat the probabilities as unverified against the reference until it
has. `examples/parity.rs` is that run: it answers a request in process and
compares every probability with a recorded server response.

```bash
KEV_DTYPE=fp32 uv run --extra serve python -m kev.serve --run jaredpalmer/kev-4b@qwen3 --port 8009
curl -s localhost:8009/v1/systemone -H 'content-type: application/json' -d @request.json > server.json

cargo run --release --example parity -- \
    --base <the base model directory> --checkpoint <the kev checkpoint> \
    --request request.json --server server.json
```

One script does the whole session, and `rkev/tests/parity/` holds the
requests it sends — five of them, each picked for something that can break: the
README's worked example, a JSON state, twelve options with two bare ones, text
that contains Kev's own delimiters, and a state past the prefix threshold. Each
goes to `/v1/systemone` and `/v1/systemone/separate`:

```bash
KEV_DTYPE=fp32 uv run --extra serve python -m kev.serve --run jaredpalmer/kev-0.6b --port 8009

scripts/parity.sh --base ~/models/qwen-qwen3-0.6b-base --checkpoint ~/models/kev-0.6b
```

It probes the server before building anything, records every answer under
`tests/parity/recordings/`, then answers the same requests in process and compares
every probability. Afterwards the recordings are files:

```bash
scripts/parity.sh --check-only --base <base> --checkpoint <kev>
```

repeats the comparison from the recordings alone, which is the point of
recording rather than checking live.

Two things it reports besides the probabilities. `input_tokens`, which is the
sharper check — probabilities can agree to four decimals over prompts that differ
by a token, token counts cannot, so a mismatch there fails the run whatever the
differences look like. And the model name each side answered as, in case the
server was serving something other than the checkpoint being compared.

Two things are checkable without a recording at all, and `examples/sanity.rs`
runs both against a real checkpoint:

```bash
cargo run --release --example sanity -- \
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
use rkev::{Forward, Pass, Result};

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

The model is the default feature. A build that only wants the wire-format types,
the errors and the `SystemOne` seam — to talk to something else, or to hold a
recording — turns it off:

```toml
rkev = { path = "../rkev", default-features = false }
```

`--features local` sits between the two: the prompt, the layout and the readout,
with `Forward` left to you.

`--features metal` adds candle's Metal backend, for an Apple GPU. It is opt-in
because it only builds on macOS, and `device("metal")` refuses without it rather
than falling back to the CPU — a silent fallback would answer the wrong question
when the reason for asking was speed:

```bash
cargo run --release --features metal --example measure -- \
    --base <base> --checkpoint <kev> --device metal
```

The precision follows the device unless `--dtype` overrides it: bf16 on a GPU, f32
on a CPU, as `kev.serve` picks it. `decide`, `sanity`, `eval` and `measure` take
`--device`; `parity` deliberately does not, since the comparison belongs on the f32
path the recording was made against.

## Quantised projections

A CPU pass is bound by how many bytes of weights it reads, so packing them into
blocks is the biggest lever there: q4k reads about a seventh of f32, q8_0 about a
quarter.

```bash
cargo run --release --example measure -- \
    --base <base> --checkpoint <kev> --quantise q8_0
```

`q4k`, `q5k`, `q6k` and `q8_0`, on `decide`, `sanity`, `eval` and `measure`.
`Backend::open_with` is the same thing from the library.

Quantising happens **after** the LoRA merge and never before: the merge stays exact
in f32 and only its result is rounded into blocks — a pre-quantised base model
could not be merged into at all. The embeddings, the norms, the convolution, the
per-head scalars and the pointer head stay dense: they are small, and the head is
where the calibration lives.

What it costs has to be measured rather than assumed, because the output is a
calibrated probability and rounding moves it. `measure --quantise` prints the time
**and** the largest difference from f32 in the same run; `eval --quantise` prints
what happened to the accuracy on your own records. Either number alone is half an
answer.

Three refusals rather than silent fallbacks: quantisation runs with **f32
activations** only (the ggml kernels want f32, and reducing the activations as well
would blend two losses that then cannot be separated), on the **CPU** only
(`QTensor::quantize` is a CPU routine, and quantised matmuls on Metal are untried
here), and a projection whose row does not divide by the block size is named in the
error together with the alternative — the k-quants pack 256 weights to a
super-block, `q8_0` packs 32.

## Measuring it on your own records

Parity asks whether this engine agrees with the reference. `examples/eval.rs` asks
the question that decides whether a checkpoint is any use to you:

```bash
cargo run --release --example eval -- \
    --base <base> --checkpoint <kev> \
    --questions questions.json --records tickets.jsonl
```

One JSON object per line, `state` plus the `labels` you consider right, keyed by
question id. A question left out of a record's labels is not scored for it, and a
label for a question that does not exist is an error rather than a silent zero —
a typo in an id would otherwise read as a perfect score on nothing.

Per question it reports the measure that fits the type (accuracy and the confusion
for a choice, accuracy plus class separation and AUC for a noul, nearest level plus
mean absolute error for a score), and then **accuracy by confidence**, which is
where a routing threshold comes from and the reason to run a model that answers
with a distribution. `--errors <n>` lists the confident mistakes, which is usually
where a question's wording is wrong rather than the model.
For CI there is a threshold: `--min-accuracy 0.8` applies to every question and
`--min-accuracy score_question=0.6` overrides one, since an ordinal scale sits
below a three-way choice and a single number for both would be either slack or
unreachable. A question below its threshold exits non-zero, and so does a question
that has a threshold and nothing labelled for it — a guarantee with no evidence
behind it is not one. `scripts/local.sh` passes the flag through.

`tests/eval/README.md` has the format, with a six-record sample beside it.

## Tests

```bash
cargo test
```

The tests check the request and response shapes against the worked example in
the Kev README, and run the backbone over a checkpoint they write themselves.
Nothing needs a server, weights or a network. Two checks want a real Qwen
tokenizer, which is not vendored here:

```bash
KEV_TOKENIZER=/path/to/tokenizer.json cargo test
```
