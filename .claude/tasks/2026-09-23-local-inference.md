# 2026-09-23 — Local inference (pure Rust, no HTTP)

Replace the HTTP round trip with an in-process Qwen + kev adapter forward pass,
keeping the existing typed API (`SystemOneRequest` -> `SystemOneResponse`).

## Why this is not mostly an inference problem

Kev generates no tokens. One forward pass, and the answer is read off the model's
hidden states. So the product is the **prompt construction plus the readout**,
not the model call. Running Qwen from Rust is the easy half; reproducing the
readout is the half that decides whether the numbers mean anything.

`jaredpalmer/kev-4b` is an *adapter* over a base model, so both have to be
loaded, or merged ahead of time.

## ANSWERED: what the readout actually is

Read from the Python kev (`kev/model.py`, `kev/api.py`, `kev/serve.py`,
`kev/mlx_model.py`, README "How It Works") on 2026-09-23. The short version, and
the reason the first plan for this work was wrong:

**The readout is not a logit readout.** A checkpoint is a rank-16 LoRA adapter
plus a small **pointer head** on the base model's backbone. The head scores each
option's `</opt>` hidden state against its question's `<decide>` hidden state,
and a softmax over those scores is the answer. The vocabulary head is never
loaded (`AutoModelForCausalLM.from_pretrained(...).model`), and no candidate
token ids exist anywhere.

1. **Prompt template** — `<state>` then, per question, `<q> instructions <opt>
   option </opt> ... <decide>`. The five delimiters are rarely-used Qwen special
   tokens (`<|fim_prefix|>`, `<|fim_middle|>`, `<|box_start|>`, `<|box_end|>`,
   `<|fim_suffix|>`), so no embedding rows are added; the adapter gives them
   their meaning. Caller text has `<|name|>` rewritten to `<¦name¦>` first, so a
   state cannot forge a delimiter. JSON states are flattened field by field,
   labels kept, two spaces per level (`kev.api.render`) — key order matters.
2. **Readout per question type** — one probability per option, from the pointer
   head. `noul` is two options (`no`, `yes`) and the answer is p(yes); `choice`
   reports the argmax and the distribution by option name; `score` reports the
   mean level index, the legend, and the distribution by level index.
3. **`confidence`** — `choice`: `(p_max - 1/K) / (1 - 1/K)`, 1.0 for a single
   option. `score`: `1 - E|level - mode| / (L - 1)`. `noul` has none.
4. **Isolation** — a block-causal mask: a token reads the state and its own
   question only, and each branch's position ids restart just after the state.
   On backbones that ignore attention masks (see below) each question runs as
   its own causal row instead, which is exact by construction.
5. **Calibration** — the head divides its scores by a temperature the checkpoint
   carries (about 2.1–2.4), which never changes the argmax.

## ANSWERED: mistral.rs cannot be the backend

Checked against the mistral.rs sources (`mistralrs-core/src/request.rs`,
`mistralrs-core/src/response.rs`, `mistralrs/src/model.rs`,
`mistralrs/src/embedding_model.rs`) on 2026-09-23. Against the five capability
questions this file asked:

1. Zero generated tokens, purely for a readout? **Yes** —
   `Model::send_raw_chat_request` runs the prompt and returns without sampling.
2. Raw logits over the whole vocabulary? **Yes** (`ResponseOk::Raw {
   logits_chunks, .. }`) — and **irrelevant**, because Kev reads hidden states,
   not logits.
3. Arbitrary positions? **No.** Only the position of the first generated token.
4. LoRA adapters? **Yes.**
5. `Send`, behind a `Mutex`? **Yes**, it is a channel handle.

The blocker is not on that list, because the list assumed the wrong readout.
What Kev needs and mistral.rs does not expose anywhere in its public API:

- the backbone's **last hidden states at chosen positions** (its embedding
  models return one pooled vector per sequence, not per position),
- a **custom additive attention mask** (the block-causal branch mask),
- **custom position ids** (each branch restarting after the state),
- a place to put the **pointer head**, which is not part of the base model.

mistral.rs is a serving engine for generation; this is not generation. Per the
rule this file set ("if 1 or 2 come back no, candle is the fallback"), the engine
is **candle** — with the caveat that candle-transformers' Qwen models also return
logits, so the backbone forward pass has to be written against candle's tensors
rather than reused.

## Done: the readout, in Rust, tested

Commit "Answer System One requests in process" — `src/prompt.rs`,
`src/encode.rs`, `src/readout.rs`, and `LocalEngine` wired up:

- The whole Kev-specific path is implemented and tested: prompt text, token
  layout, mask, positions, pointer head, softmax, the three answer shapes, both
  confidence formulas, `usage`, and `Error::ContextOverflow` for a question that
  does not fit (which reports `is_validation`, as the server's 422 does).
- `Forward` was re-cut to what the readout needs: hidden states at the readout
  positions, given a `Pass` (ids, position ids, segments, readout positions).
  `Pass::attends` is the mask; `Pass::rows` splits a pass into one causal row per
  question for backbones that cannot honour it.
- `tests/local_engine.rs` drives all of it through a stub backend whose hidden
  states are constructed to yield a chosen distribution, so the Kev README's
  worked example comes back with its published numbers (choice confidence 0.205
  which the README prints as 0.21, score 1.44, score confidence 0.78). Packed
  and separate calls are asserted to agree, as is the row form.

Not implemented, deliberately: `option_isolation` (an ablation the released
checkpoints do not serve with), the state-prefix cache, and `permute`.

## Left to do

1. **The backbone.** A candle Qwen3.5 (and Qwen3) forward pass that takes the
   mask and position ids and returns hidden states, with the LoRA adapter merged
   and `head.pt`'s two projections plus temperature loaded. Qwen3.5 mixes
   attention with Gated DeltaNet layers, which ignore masks — so the row form is
   mandatory there, and a DeltaNet implementation is needed. The Qwen3
   generation (`jaredpalmer/kev-4b@qwen3`, attention-only) is the cheaper first
   target.
2. **The tokenizer.** The `tokenizers` crate with the checkpoint's
   `tokenizer.json`, mapping the five delimiters to ids.
3. **`head.pt`.** A torch pickle; either read it directly or convert it to
   safetensors ahead of time.
4. **Parity.** Run the same requests against a live server and against the
   engine, assert every probability matches within a tolerance, and record the
   server's answers as fixtures. This is the only test that proves the readout
   against the real model; everything above proves it against its own spec.

## Design decisions

- **Engine**: mistral.rs is ruled out (above). candle, with our own forward pass.
- **Sync vs async**: DECIDED — the engine is blocking inside, `spawn_blocking`
  at the `SystemOne` seam.
- **Trait seam**: `impl Future + Send` (RPITIT), no `async_trait`. Consequence:
  not dyn-compatible; a runtime-chosen backend needs an enum.
- **Features**: `http` (reqwest) and `local` are independent; the types, the
  errors and the `SystemOne` trait build with neither.

## Status

- PR 1 (seam) and PR 2 (`http` feature) done, and verified on 2026-09-23 on
  cargo 1.97.1 on macOS.
- PR 3 (engine skeleton) done and verified the same day.
- PRs 4–7 (prompt, `noul`/`choice`/`score` readout, confidence) done in one
  commit: they are one behaviour, and split up they would have been dead code.
- PR 8: `system_one_separate` is done (one pass per question, as the server does
  it). `permute` is not.
- Next: the candle backbone, then the parity run against a live server.

## Environment note

Earlier in this container: no Rust toolchain and no network. Both changed — the
network came back (GitHub, crates.io, static.rust-lang.org), so rustup installs,
and the sources above could be read instead of guessed.

There is still **no C toolchain** (no `cc`, no glibc `crt1.o`), so the default
`cargo test` cannot link. What works here, and how everything above was run:

```bash
# once
rustup target add aarch64-unknown-linux-musl
mkdir -p ~/lib-shim && cd /usr/lib/aarch64-linux-gnu \
  && ln -sf $PWD/libc.so.6 ~/lib-shim/libc.so \
  && ln -sf $PWD/libm.so.6 ~/lib-shim/libm.so \
  && ln -sf $PWD/libgcc_s.so.1 ~/lib-shim/libgcc_s.so \
  && for l in dl pthread rt util; do ln -sf $PWD/libc.so.6 ~/lib-shim/lib$l.so; done
# ~/bin/rustc-shim: build scripts -> musl (static, still runs here);
# proc-macro dylibs -> rust-lld against ~/lib-shim; flags appended last so
# cargo clippy also works.
RUSTC_WRAPPER=~/bin/rustc-shim cargo test --target aarch64-unknown-linux-musl \
    --no-default-features --features local
```

Verified this way on 2026-09-23 (cargo 1.98.1): `--no-default-features` and
`--features local` green — 15 engine tests, 2 seam tests, 10 wire-format tests,
1 doctest — plus `cargo fmt --check` and `cargo clippy --all-targets` clean.

The `http` feature cannot be built here at all: `ring`'s build script needs a
real C compiler. Anything HTTP-side stays verified only on the Mac.
