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

**The conventions are verified against Hugging Face**, which was the open
question: `tests/qwen3.rs` carries a plain-f32 transcription of
`transformers/models/qwen3/modeling_qwen3.py` (RMSNorm, the per-head query and
key norms before the rotary embedding, `rotate_half` with
`inv_freq[i] = theta^(-2i/d)`, grouped-query head mapping, `head_dim ** -0.5`,
SwiGLU, residual order) and asserts every hidden unit of every token agrees with
the candle path to 1e-5. They agree to about 1e-6, which is f32 accumulation
order. There is a second, narrower test on the rotary tables alone.

Both were checked for sensitivity by breaking the implementation on purpose: the
interleaved rotary convention (candle's `rope_i`, which runs and looks fine)
moves a hidden unit by 9e-2, and rotating before the per-head norm rather than
after moves one by 5e-3. Neither is caught by any other test in the suite, which
is the argument for having this one.

**The tokenizer is verified against transformers**, at the source and by test.
On the Python side `kev.model.user_tokens` is
`tok(re.sub(r"<\|([A-Za-z0-9_]+)\|>", r"<¦\1¦>", text), add_special_tokens=False)`,
and in transformers 5 (`tokenization_utils_tokenizers.py`,
`TokenizersBackend._encode_plus`) that call reduces to the Rust `tokenizers`
crate — the same library this crate uses — as
`encode_batch(texts, add_special_tokens=False, is_pretokenized=False)`, with
`no_truncation()` and `no_padding()` applied on every call and
`encode_special_tokens` left false. So:

- Truncation and padding are now switched off explicitly when a tokenizer is
  loaded. They would otherwise be inherited from `tokenizer.json`, and that is
  a difference in the token ids. This was a real gap, not a formality.
- `encode_special_tokens = false` matches the Python default, which means a
  special token written out in the text is *matched*, not split — the hazard the
  escaping exists for. It is set explicitly too, so a future default cannot move
  it quietly.
- The escaping is pinned against the regex's semantics, including the cases that
  must **not** match (`<||>`, `<|a-b|>`, `<|Ünicode|>`, `<|abc|def|>`). The
  scanner does not backtrack and does not need to: a name is a maximal run of
  `[A-Za-z0-9_]`, which cannot contain `|`, so no shorter name can be followed
  by one either.
- `tests/qwen3.rs` proves the forgery is prevented where it matters, with the
  delimiters as added tokens exactly as Qwen ships them: a state containing
  `<|fim_middle|>` tokenises to the delimiter id on its own, and cannot produce
  it through the engine.

Checked for sensitivity the same way as the backbone: with the escaping removed,
those tests fail and say which delimiter was forged.

**`head.pt` is read whole.** `pointer_head()` takes the two projections *and*
the temperature the checkpoint was calibrated with; nothing is passed in. Two
things had to be learned from candle's pickle reader to get there:

- The tensors sit under `head` in `kev.checkpoint.Meta`, so the reader has to be
  given that key. Without it, it walks the metadata instead and finds **no
  tensors at all** — silently, which is how this was nearly missed.
- A temperature is a float, and tensor readers return tensors, so the pickle is
  walked for it (candle exposes its pickle machine, which is enough). Shallower
  matches win, so a checkpoint's own value beats anything nested.

A safetensors head carries no temperature and is left raw, which is also what
`Meta.temperature` defaults to. The tests build a `head.pt` in the shape
`torch.save` writes — zip, protocol-2 pickle, an `OrderedDict` state dict, one
zip entry per storage — and assert the loaded head's logits equal a hand-built
one's divided by the temperature. Drop the `head` key and that test fails with
"it holds []".

**Still not verified against real weights.** Nothing here has opened a Qwen3
checkpoint, so what remains open is the loading rather than the arithmetic: the
tensor names and shapes of a published base (a mismatch fails loudly, at least),
a real `tokenizer.json` (`KEV_TOKENIZER=<path> cargo test --features qwen3` runs
the delimiter and forgery checks against one; huggingface.co is not reachable
from this container, so it was exercised against a file in Qwen's shape instead),
and the probabilities end to end.

## Left to do

1. **Parity — the tool is there, it has not been run.** `examples/parity.rs`
   answers a recorded request in process and compares every probability with a
   recorded server response (`--tolerance`, non-zero exit past it). What is left
   is running it on a machine that can hold `jaredpalmer/kev-4b@qwen3` (or
   `kev-0.6b`, which is small) with `KEV_DTYPE=fp32` on the server side, and
   turning the recordings into offline fixtures afterwards. The arithmetic and the
   tokenizer call are now checked against Hugging Face's definitions of them;
   this is what checks the loading and the real vocabulary against the real
   thing. Run it with `KEV_TOKENIZER` set too, so the opt-in tokenizer checks
   come along. A real `head.pt` is part of what it exercises: the fixture here is
   in torch's shape, but only torch writes the real thing.
2. **The Qwen3.5 bases** (the current checkpoints): Gated DeltaNet layers, which
   means a second backbone and the row form only. mistral.rs has a candle
   implementation of that architecture in `models/qwen3_next.rs` (MIT), which is
   worth reading before writing one.
3. **Performance.** No KV cache, no state-prefix reuse, CPU by default, f32. A
   repeated state pays for itself every time; the Python caches it.
4. `permute`, and `option_isolation` if a checkpoint ever serves with it.

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
