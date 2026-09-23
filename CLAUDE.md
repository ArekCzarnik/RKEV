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
- `src/local.rs` — `LocalEngine` (feature `local`), the `Forward` trait and
  `Pass`. The engine owns everything Kev-specific and leaves a backend one job:
  run the backbone, return hidden states at the readout positions. **Not**
  logits — a checkpoint's output layer is the pointer head, which is why a
  text-generation API such as mistral.rs cannot serve as the backend at all; the
  task file has the evidence. Three load-bearing details: `Clone` is cheap
  because `spawn_blocking` needs `'static`, the backend lives in an
  `Arc<Mutex<dyn Forward>>` so clones share one loaded model, and `Pass::rows`
  exists because backbones with recurrent layers (Qwen3.5, so every current
  checkpoint) cannot honour `Pass::attends`. No backend ships yet.
- `src/error.rs` — `Error` with predicates (`is_validation` for Kev's 422,
  `is_unauthorized` for 401/403) instead of making callers match on status
  codes.

### Features

`http` (on by default) pulls in `reqwest`; `local` pulls in tokio for
`spawn_blocking` and will carry the inference backend once one is written. The
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

### Tests

Three files, all offline. `tests/wire_format.rs` pins serialised requests and
deserialised responses against the worked example in the Kev README, so a
refactor cannot silently change what goes on the wire. `tests/seam.rs` proves
the `SystemOne` trait is implementable without reqwest. `tests/local_engine.rs`
drives the engine over a backend with no model in it, whose hidden states are
built to produce a chosen distribution — that is what makes the prompt, the
question isolation and the readout testable without weights, and it is where the
Kev README's published numbers are asserted. Tests
assert on JSON shape and public accessor behaviour, never on internals — keep
new tests in that style, and update the README example and these fixtures
together whenever the wire format legitimately changes.

## Documentation

The crate is documented in three places that must stay in sync when the public
API changes: `kev-client/README.md`, the `//!` module docs in `src/lib.rs`
(a compiled `no_run` doctest), and `examples/triage.rs`. All three use the same
support-ticket triage example as the README of the upstream Kev project.
