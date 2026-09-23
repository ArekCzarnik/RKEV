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

## Done: the Qwen3 backbone, in candle

Commit "Run the attention-only Qwen3 bases in candle" — `src/qwen3.rs`,
`src/backend.rs`, feature `qwen3`:

- The attention-only Qwen3 forward pass: embeddings, RMSNorm, GQA with Qwen3's
  per-head query/key norms, rotary embeddings at **the position ids we hand it**,
  SwiGLU, final norm. No vocabulary head and no KV cache — one prefill pass is a
  whole request. f32 throughout, the precision the published numbers use.
- The LoRA adapter is merged as the weights are read, `W + (B @ A) * alpha / r`
  in f32, exactly as `LoadOptions.merge` does it. `use_rslora` is honoured;
  adapters with trained token embeddings are refused.
- `Qwen3Backend` adds the tokenizer (`tokenizers`, the checkpoint's
  `tokenizer.json`) and implements `Forward`. `pointer_head()` reads the head
  from `head.pt` or a safetensors file.
- Configs this backbone would answer wrongly are refused: hybrid layer types,
  sliding-window attention, attention bias, heads that do not divide.

`tests/qwen3.rs` writes a two-layer checkpoint (config, safetensors, tokenizer,
head — the safetensors writer is 30 lines in the test, so no dev-dependency) and
runs the real thing over it. The weights are noise, so the tests assert only
what holds for any weights, which is also what the Python asserts:

- a question's answer does not move when another question changes (the mask),
- the packed pass and `Pass::rows` agree (`test_rows_match_packed`),
- packed and separate calls agree,
- an adapter merged at load time equals a checkpoint that ships it pre-merged,
- a hybrid config is refused.

**Not verified against real weights.** Nothing here has seen a Qwen3 checkpoint,
so the numerics against HF transformers are still open — the likely places for a
mistake are the rotary convention and the order of the per-head norms.

## Left to do

1. **Parity.** Load `jaredpalmer/kev-4b@qwen3` (or `kev-0.6b`, which is small),
   run the same requests against a live `kev.serve` and against the engine, and
   assert every probability matches within a tolerance. Record the server's
   answers as fixtures afterwards so it runs offline. This is the step that
   turns the backbone from plausible into correct.
2. **The temperature.** `pointer_head()` takes it as an argument because the
   tensor readers only return tensors; reading it out of `head.pt` itself needs
   a little pickle work. Until then a caller has to pass the checkpoint's value
   (`/v1/models` reports it), and passing the wrong one silently miscalibrates
   every probability without changing any argmax.
3. **The Qwen3.5 bases** (the current checkpoints): Gated DeltaNet layers, which
   means a second backbone and the row form only. mistral.rs has a candle
   implementation of that architecture in `models/qwen3_next.rs` (MIT), which is
   worth reading before writing one.
4. **Performance.** No KV cache, no state-prefix reuse, CPU by default, f32. A
   repeated state pays for itself every time; the Python caches it.
5. `permute`, and `option_isolation` if a checkpoint ever serves with it.

## Design decisions

- **Engine**: mistral.rs is ruled out (above). candle, with our own forward pass
  — pinned to 0.9, because 0.10 made `candle-core` depend on `tokenizers` with
  oniguruma, a C dependency that buys this crate nothing.
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
- The attention-only Qwen3 backbone is done and tested on a synthetic
  checkpoint. Next: the parity run against a live server, which is what decides
  whether it is right.

## Environment note

Earlier in this container: no Rust toolchain and no network. Both changed — the
network came back (GitHub, crates.io, static.rust-lang.org), so rustup installs,
and the sources above could be read instead of guessed.

There is still **no C toolchain**: no `cc`, and no glibc `crt1.o`/`libc.so`,
though glibc itself is there to run against. Three small things make
`cargo test` work anyway, all of them local to this machine and none of them in
the repo:

```bash
# 1. point the linker at the runtime libraries cc would have found
mkdir -p ~/lib-shim && cd /usr/lib/aarch64-linux-gnu \
  && ln -sf $PWD/libc.so.6 ~/lib-shim/libc.so \
  && ln -sf $PWD/libm.so.6 ~/lib-shim/libm.so \
  && ln -sf $PWD/libgcc_s.so.1 ~/lib-shim/libgcc_s.so \
  && for l in dl pthread rt util; do ln -sf $PWD/libc.so.6 ~/lib-shim/lib$l.so; done

# 2. supply the startup object: ~/crt/crt1.rs is glibc's aarch64 start.S as a
#    global_asm! block, built with
#    rustc --crate-type lib --emit=obj -O ~/crt/crt1.rs -o ~/crt/crt1.o

# 3. ~/bin/rustc-shim adds, to every rustc call:
#      -Clinker=<toolchain>/rust-lld -Lnative=~/lib-shim -Clink-arg=~/crt/crt1.o
#      -Clink-arg=-dynamic-linker -Clink-arg=/lib/ld-linux-aarch64.so.1
#    (the last one is not optional: without PT_INTERP the binary segfaults
#     before main, and nothing says why)

RUSTFLAGS=-Ctarget-feature=+fp16 RUSTC_WRAPPER=~/bin/rustc-shim \
    cargo test --no-default-features --features qwen3
```

`+fp16` is for `gemm-f16`, whose inline assembly does not build on this
aarch64 target without it; a Mac has it in the base feature set.

Verified this way on 2026-09-23 (cargo 1.98.1): `--no-default-features`,
`--features local` and `--features qwen3` all green — 6 backbone tests, 15
engine tests, 2 seam tests, 10 wire-format tests, 1 doctest — with
`cargo fmt --check` and `cargo clippy --all-targets` clean.

The `http` feature still cannot be built here: `ring`'s build script wants a
real C compiler. Anything HTTP-side stays verified only on the Mac.
