# 2026-09-23 — Local inference (pure Rust, no HTTP)

Replace the HTTP round trip with an in-process Qwen + kev adapter forward pass,
keeping the existing typed API (`SystemOneRequest` -> `SystemOneResponse`).

## Why this is not mostly an inference problem

Kev generates no tokens (`src/types.rs:218`). One forward pass, probabilities
read straight off the logits. So the product is the **prompt construction plus
the logit readout**, not the model call. Running Qwen from Rust is the easy
half; reproducing the readout bit-for-bit is the half that decides whether the
numbers mean anything.

`jaredpalmer/kev-4b` is an *adapter* over a base model (README: "downloads the
adapter and the base model"), so both have to be loaded, or merged ahead of time.

## Blocked on — mistral.rs capability

Kev never generates. It needs the probabilities of specific candidate tokens,
which means a forward pass whose logits can be read. Before writing the
`Forward` impl, confirm in the mistral.rs docs or source:

1. Can a request be run for **zero generated tokens**, purely for logits?
2. Is there access to the **raw logits vector** over the whole vocabulary, or
   only top-k logprobs from the sampler? With up to 255 choice options, top-k
   will not cover the candidates unless k can be set very high.
3. Logits at **one position only (the last), or at arbitrary positions**? This
   decides whether `Forward::logits` can keep its `positions` argument or has
   to be one call per position — and it interacts with readout item 2 below.
4. How a **LoRA adapter** is loaded (merged ahead of time, or at runtime).
5. Whether the model handle is `Send` and can live behind a `Mutex` the way
   `LocalEngine` assumes.

If 1 or 2 come back "no", mistral.rs is the wrong engine for this and candle
(direct tensor access, since it is candle underneath anyway) is the fallback.

## Blocked on — needed from the Python Kev repo

Nothing in this repo describes the readout. Required before any engine work:

1. **Prompt template** — how `state` + `questions` become one prompt string,
   including how a JSON object/array state is "converted to labelled text",
   and in what order questions are laid out.
2. **Readout per question type** — which token position(s) are read, which
   vocabulary ids, and how they are normalised:
   - `noul`  -> probability of yes
   - `choice` -> one probability per option name
   - `score`  -> distribution over level indices, and how the mean `score` falls out
3. **`confidence`** — the exact formula (docs call it "a shape measure of the
   distribution", `src/types.rs:298`).
4. **Isolation** — how questions are kept from reading each other inside one
   forward pass. This constrains the prompt/attention layout and is what
   `/v1/systemone/separate` exists to verify.
5. **Base model id + chat template / special tokens**, and the tokenizer used.

## Design decisions still open

- **Engine**: DECIDED - mistral.rs. It sits behind the `Forward` trait in
  `src/local.rs`, so candle stays a one-file swap if the capability question
  below goes the wrong way.
- **Sync vs async**: a forward pass is CPU/GPU-bound synchronous work. The
  current API is `async` (`src/client.rs:79`). Either the local engine gets
  wrapped in `spawn_blocking`, or the trait stays sync and the HTTP client
  keeps its own async surface. Decide before writing the trait.
- **Trait seam**: `system_one` / `system_one_separate` are the natural cut.
  Note Rust 1.75 (`Cargo.toml`) has async-fn-in-trait but no `Send` bound on
  the returned future — needs RPITIT with an explicit `+ Send`, or `async_trait`.
- **Features**: make `reqwest` optional (`http` feature) so a local-only build
  does not pull the HTTP stack; `local` feature for the engine.

## Parity strategy

There is a reference implementation, so use it: run the same request against a
live Kev server and against the local engine, assert every probability matches
within a tolerance. Record server responses as fixtures so the comparison runs
offline afterwards. This is the only test that actually proves the readout;
shape tests (`tests/wire_format.rs`) cannot.

## Proposed PR sequence

1. Trait seam + move the HTTP client behind it. No behaviour change.
2. `reqwest` behind an `http` feature.
3. Engine skeleton: load base + adapter, one forward pass, raw logits out. No
   Kev semantics yet.
4. `noul` readout + parity test against recorded server output.
5. `choice` readout + parity test.
6. `score` readout (+ `legend`, mean) + parity test.
7. `confidence` + parity test.
8. Local `system_one_separate`; local `permute` (option-order shuffling).

## Status

- PR 1 done: `SystemOne` trait in `src/client.rs`, `Client`
  implements it, inherent methods kept so no caller changes. Path literals
  pulled into `SYSTEM_ONE_PATH` / `SYSTEM_ONE_SEPARATE_PATH`. Seam tests in
  `tests/seam.rs`.
- Decision taken: `impl Future + Send` (RPITIT), no `async_trait` dependency.
  Consequence: not dyn-compatible. A runtime-chosen backend needs an enum,
  not `Box<dyn SystemOne>`. Revisit in PR 3 if that hurts.
- PR 2 done: `reqwest` optional behind a default-on `http`
  feature. Trait moved to `src/system_one.rs` so the seam survives without it;
  `Error::Transport` + `From<reqwest::Error>` cfg-gated; crate doctest body and
  the triage example gated too. `tests/seam.rs` gained a non-HTTP `Canned`
  backend that proves the trait is implementable without reqwest.
- PR 1 and PR 2 VERIFIED on 2026-09-23: cargo 1.97.1 on macOS, 10 tests + the
  doctest green with default features, and `--no-default-features` green too.
  The RPITIT trait, the cfg-gated error arms and the feature split all hold.
- PR 3 skeleton VERIFIED on 2026-09-23: `local` feature, `Error::Engine`, the
  `Forward` trait + `LocalEngine` in `src/local.rs`, `tests/local_engine.rs`
  driving it through a stub backend, feature matrix in `scripts/test.sh`.
  Clippy clean and all four feature combinations green on cargo 1.97.1.
  Two fixes were needed on the way: `MutexGuard<'_, dyn Forward + 'static>`
  (invariance over T), and `#[allow(clippy::manual_async_fn)]` on the impls
  that return a bare async block. Clippy now runs with `--all-features`,
  without which lints inside cfg-gated modules are never seen.
- Engine decided: mistral.rs. Its `Forward` impl is NOT written. Writing calls
  against an API that cannot be compiled or looked up here would be invention,
  not code - answer the five capability questions above first.

## Environment note

This container has no Rust toolchain and no network (verified: rustup, crates
and github all unreachable). Nothing here has been compiled or test-run. Any
Rust written in this state is unverified until built on a machine with cargo.
