# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Layout

The repo root is not a Cargo workspace. The only crate is `kev-client/`, so all
`cargo` commands must be run from inside that directory.

## Commands

`scripts/test.sh` wraps the whole sequence (fmt check, clippy, tests) and can
be run from any directory; arguments are passed through to `cargo test`:

```bash
scripts/test.sh                                     # everything
scripts/test.sh request_matches_the_readme_example  # one test by name
scripts/test.sh --skip-fmt --skip-clippy            # tests only
```

Or directly:

```bash
cd kev-client
cargo test                                   # all tests; needs no server
cargo test --test wire_format                # the integration test file
cargo test request_matches_the_readme_example  # a single test by name
cargo run --example triage                   # needs a running Kev server
cargo run --example triage -- "custom ticket text"
cargo doc --open                             # the crate is documented in rustdoc
```

**Never report a `cargo test` result you did not run** — say plainly that a
change is unverified instead.

This container has no C toolchain (no `cc`, no glibc `crt1.o`), so a plain
`cargo test` cannot link, and `--features http` cannot be built here at all
(`ring` needs a C compiler). A Rust toolchain installs with `rustup`, and the
non-HTTP features do run here through the musl target; the recipe is at the end
of `.claude/tasks/2026-09-23-local-inference.md`.

`scripts/triage.sh` runs the example; it probes the server first and passes
`KEV_*` through:

```bash
scripts/triage.sh                              # the README ticket
scripts/triage.sh "My parcel never arrived."   # your own ticket
scripts/triage.sh -f ticket.txt                # or from a file / stdin
```

`cargo run --example triage` reads `KEV_BASE_URL`, `KEV_MODEL` and
`KEV_API_KEY`; without them it targets `http://127.0.0.1:8009` with model
`kev-latest` and no auth. Starting a server needs the separate Python Kev repo
(`uv run --extra serve python -m kev.serve --run jaredpalmer/kev-4b --port 8009`),
which is not vendored here.

## Architecture

`kev-client` is a thin async HTTP client for [Kev](https://github.com/jaredpalmer/kev),
which implements TypeSafe's System One API. One call sends a **state** (any
text or JSON document) plus a map of **questions**; the server answers each
question with a probability distribution rather than a single label. Questions
share the state but cannot read each other.

Three modules behind a flat re-export surface in `src/lib.rs`:

- `src/types.rs` — the wire format. `SystemOneRequest` is built fluently
  (`.ask(id, question)`); `Question` is an internally tagged enum
  (`type: noul | choice | score`) that the `Noul`/`Choice`/`Score` builders
  convert into via `From`. `Answer` is the mirror-image tagged enum on the
  response side, with accessors (`as_noul`, `as_choice`, `as_score`,
  `confidence`, `probabilities`, `top`, `legend`) that return `Option` rather
  than panicking on a type mismatch.
- `src/system_one.rs` — the `SystemOne` trait: the backend seam. HTTP today, a
  local inference engine later. Deliberately outside `client.rs` so it stays
  available with the `http` feature off.
- `src/client.rs` — `Client` (feature `http`) and the shared `read()` helper
  that every request funnels through: it captures the `x-typesafe-request-id` header, turns non-2xx
  into `Error::Api`, and reads the body as text before decoding so a failed
  parse can report what actually arrived. `with_model_filled_in` injects the
  client's default model only when the request did not pin one.
- `src/prompt.rs`, `src/encode.rs`, `src/readout.rs` (feature `local`) — the
  Kev-specific half of a local forward pass, mirroring `kev/api.py` and
  `kev/model.py`: the text the model sees, the token layout (`<state>`, then
  `<q> instructions <opt> option </opt> ... <decide>` per question) with its
  block-causal mask and per-branch positions, and the pointer head that scores
  each `</opt>` hidden state against its question's `<decide>` and turns the
  softmax into an `Answer`. Every rule here is copied from the Python rather
  than invented; when one changes, the answers change.
- `pointer_head()` in `src/backend.rs` loads the other half of a checkpoint from
  `head.pt`: the projections, which sit under the `head` key (a reader not told
  that finds no tensors and says nothing), and the calibration temperature,
  which is a float and so has to be walked out of the pickle by hand.
- `src/local.rs` — `LocalEngine` (feature `local`), the `Forward` trait and
  `Pass`. The engine owns everything Kev-specific and leaves a backend one job:
  run the backbone, return hidden states at the readout positions. **Not**
  logits — a checkpoint's output layer is the pointer head, which is why a
  text-generation API such as mistral.rs cannot serve as the backend at all; the
  task file has the evidence. Three load-bearing details: `Clone` is cheap
  because `spawn_blocking` needs `'static`, the backend lives in an
  `Arc<Mutex<dyn Forward>>` so clones share one loaded model, and `Pass::rows`
  exists because backbones with recurrent layers (Qwen3.5) cannot honour
  `Pass::attends`.
- `src/qwen3.rs`, `src/qwen3_5.rs`, `src/weights.rs`, `src/backend.rs` (feature
  `candle`) — the model itself, in candle. Both backbones exist separately from
  candle's and mistral.rs' Qwen3 because they have to take an arbitrary additive
  mask and explicit position ids, and have to stop at the hidden states: no
  vocabulary head, no KV cache, one prefill pass. `weights.rs` reads a checkpoint
  and merges its LoRA in f32 as it goes; `backend.rs` puts a backbone and the
  tokenizer together as `Backend`, and picks the backbone from `config.json`.
  `qwen3_5.rs` is the current generation and the harder one: three quarters of its
  layers are a gated delta rule (a recurrence, not attention), the rest is
  attention with an output gate and only a quarter of each head rotated, and
  every norm is zero-centred (`x * (1 + w)`) except the recurrence's gated one.
  **A recurrent layer cannot honour `Pass::attends`**, so a hybrid checkpoint
  answers one causal row per question — exact isolation instead of masked, and the
  same thing the Python does there.
- `src/error.rs` — `Error` with predicates (`is_validation` for Kev's 422,
  `is_unauthorized` for 401/403) instead of making callers match on status
  codes.

### Features

`http` (on by default) pulls in `reqwest`; `local` pulls in tokio for
`spawn_blocking`; `qwen3` adds `local` plus candle and tokenizers, i.e. the
actual model. candle is pinned to 0.9 on purpose: 0.10 made `candle-core` depend
on `tokenizers` with oniguruma, a C dependency this crate has no use for. The
types, the errors and the `SystemOne` trait build with neither, so a
local-inference build carries no HTTP stack. Consequences to keep in mind when editing:
`Error::Transport` and `From<reqwest::Error>` are `#[cfg]`-gated, the crate-level
doctest hides its body behind `#[cfg(feature = "http")]`, and the triage example
declares `required-features = ["http"]`. `scripts/test.sh` runs
a feature matrix (`--no-default-features`, `--features local`, `--all-features`)
so the split cannot rot.

### Invariants worth preserving

- **Order is semantic.** Question order and `choice` option order are part of
  the request and can move the model's answer, so every map on the wire is an
  `IndexMap`, never a `HashMap` or `BTreeMap`. `indexmap::IndexMap` is
  re-exported from the crate root because it leaks into the public API.
- **`Noul` has no confidence.** For a yes/no question the probability *is* the
  answer, so `confidence()` and `probabilities()` deliberately return `None`.
  Confidence elsewhere is a shape measure of the distribution, not an accuracy
  rate — keep the docs saying so.
- **Undocumented endpoints stay `serde_json::Value`.** `permute` and `models`
  return raw JSON on purpose: the API docs do not pin their envelopes, and a
  typed struct would invent a contract. Do not "improve" these into structs.
- **Optional response fields.** Only `model` and `answers` are guaranteed;
  `usage` and `latency_ms` use `#[serde(default)]`. `request_id` is
  `#[serde(skip)]` — it comes from a header, not the body.
- **`serde_json` keeps `preserve_order` on.** A JSON state is rendered into the
  prompt field by field, so sorted keys would build a different prompt — and a
  different answer — than the server does. It is not there for convenience.
- **The local engine is a port, not a design.** `prompt.rs`, `encode.rs` and
  `readout.rs` reproduce the Python kev, down to `True` for a boolean and
  `json.dumps`' spacing in `usage.output_tokens`. Change them only to follow the
  Python, and name the function upstream when you do.
- **Delimiters are unforgeable, and that is load-bearing.** Caller text has
  `<|name|>` rewritten to `<¦name¦>` before tokenising, because a tokenizer
  matches its own special tokens inside ordinary text (`encode_special_tokens`
  is false in transformers and here). Without it a state could open a question.
  `Backend` also switches truncation and padding off explicitly, as
  transformers does on every call — inheriting them from `tokenizer.json` would
  change the token ids.

### Tests

Three files, all offline. `tests/wire_format.rs` pins serialised requests and
deserialised responses against the worked example in the Kev README, so a
refactor cannot silently change what goes on the wire. `tests/seam.rs` proves
the `SystemOne` trait is implementable without reqwest. `tests/local_engine.rs`
drives the engine over a backend with no model in it, whose hidden states are
built to produce a chosen distribution — that is what makes the prompt, the
question isolation and the readout testable without weights, and it is where the
Kev README's published numbers are asserted. `tests/qwen3.rs` writes a
two-layer checkpoint (config, safetensors, tokenizer, head) into the temp
directory and runs the real backbone over it: made-up weights, but the
properties under test hold for any weights — question isolation, packed against
row form, and that merging an adapter equals a pre-merged checkpoint. It also
carries a plain-f32 transcription of `modeling_qwen3.py` and asserts the candle
path matches it everywhere, which is what pins the conventions Hugging Face and
candle disagree about (the rotary halves above all); keep those tests honest by
breaking the implementation on purpose when you touch them. It also holds the
tokenizer checks, one of which runs only with `KEV_TOKENIZER=<tokenizer.json>`
set, since a real Qwen tokenizer cannot be vendored. `tests/qwen3_5.rs` does the
same for the hybrid backbone, transcription included — the delta rule has a decay,
a write strength, an L2 norm (not an RMS norm), a short convolution and a gated
output norm, and every one of them is a place to be quietly wrong.
`tests/fixtures/mod.rs` is shared by both: the safetensors and tokenizer writers,
so no test needs a dev-dependency to build a checkpoint. Tests
assert on JSON shape and public accessor behaviour, never on internals — keep
new tests in that style, and update the README example and these fixtures
together whenever the wire format legitimately changes.

### The state prefix

Every question of a request shares the state, so the state is run once and each
question continues from it: attention layers keep the state's keys and values, a
recurrent layer keeps its state matrix and the tail of its convolution window.
Exact, because the state comes first and neither layer kind looks forward — the
same reuse `kev.serve` does, for the same reason. `Backend` also keeps the last
few states across requests, keyed by their token ids.

The defaults differ by backbone, and the reason is measured, not assumed
(`cargo test --features candle -- --ignored --nocapture` prints it):

- A **recurrent** base would otherwise run the whole state through every layer
  once per question, so the reuse always pays: 57 ms to 24 ms for five questions
  on a 241-token state, and 14 ms when the state is a cache hit.
- An **attention-only** base already runs the state once in the packed pass. A
  cache *miss* is then a few percent slower (several small passes instead of one
  big one), a *hit* skips the state altogether: on a 1200-token state, 129 ms to
  20 ms. So the prefix path starts at 384 state tokens there, which is what
  `kev.serve` uses too.

`Backend::with_prefix(false)` turns it off and takes the packed path. That is the
path the transcription tests compare against, so leave those calling it.

### Checking the engine against the server

`examples/parity.rs` (feature `candle`) answers a recorded request in process and
compares every probability with the server's recorded response, exiting non-zero
past a tolerance. It needs no HTTP stack on purpose: the server side is a file,
which makes the recording reusable as an offline fixture later. Nothing else in
this repo compares the engine with the reference *running*, so this is the check
that decides whether the local numbers mean anything.

## Documentation

The crate is documented in three places that must stay in sync when the public
API changes: `kev-client/README.md`, the `//!` module docs in `src/lib.rs`
(a compiled `no_run` doctest), and `examples/triage.rs`. All three use the same
support-ticket triage example as the README of the upstream Kev project.
