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
`src/backend.rs`, feature `candle`:

- The attention-only Qwen3 forward pass: embeddings, RMSNorm, GQA with Qwen3's
  per-head query/key norms, rotary embeddings at **the position ids we hand it**,
  SwiGLU, final norm. No vocabulary head and no KV cache — one prefill pass is a
  whole request. f32 throughout, the precision the published numbers use.
- The LoRA adapter is merged as the weights are read, `W + (B @ A) * alpha / r`
  in f32, exactly as `LoadOptions.merge` does it. `use_rslora` is honoured;
  adapters with trained token embeddings are refused.
- `Backend` adds the tokenizer (`tokenizers`, the checkpoint's
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
a real `tokenizer.json` (`KEV_TOKENIZER=<path> cargo test --features candle` runs
the delimiter and forgery checks against one; huggingface.co is not reachable
from this container, so it was exercised against a file in Qwen's shape instead),
and the probabilities end to end.

## Done: the Qwen3.5 backbone, Gated DeltaNet and all

Commit "Run the Qwen3.5 bases: attention mixed with Gated DeltaNet" —
`src/qwen3_5.rs`, with the shared loading moved to `src/weights.rs` and `Backend`
picking the backbone from `config.json`. This is the generation the *current*
checkpoints use, so it is the one that matters.

What it is, from `modeling_qwen3_5.py`:

- Three quarters of the layers are a **gated delta rule**: a short depthwise
  convolution over time, then a recurrence with one state per value head —
  `S <- S exp(g_t)`, `delta <- (v_t - k_t S) beta_t`, `S <- S + k_t^T delta`,
  `out_t <- q_t S` — where `beta = sigmoid(b)` is the write strength and
  `g = -exp(A_log) softplus(a + dt_bias)` the decay. Queries and keys are
  **L2-normalised** (not RMS), the query is scaled by the *key* width, key heads
  are shared by several value heads (repeat_interleave, so value head `h` reads
  key head `h / group`), and the output goes through a **ones-centred gated** RMS
  norm multiplied by `silu(z)`.
- The attention layers are Qwen3's plus two twists: `q_proj` is twice as wide and
  its second half is a **sigmoid output gate** applied before `o_proj`, and only
  `partial_rotary_factor` of each head is rotated (0.25 by default, the rest
  passes through unchanged).
- Every norm is **zero-centred** (`x * (1 + w)`, from a parameter initialised to
  zeros), which the loader folds into a single multiplication. The recurrence's
  gated norm is the one exception.
- The recurrence runs sequentially, one token at a time: exact, and slow. The
  chunked form in the reference computes the same thing faster.

Because a recurrence cannot honour an attention mask, a hybrid checkpoint answers
**one causal row per question** (`Pass::rows`), which is what the Python does
there too. Isolation is then exact by construction, and `tests/qwen3_5.rs`
asserts it as equality rather than within a tolerance.

That test carries a transcription of the whole layer stack, recurrence included,
and compares every hidden unit of every token. Five deliberate mistakes were
tried against it, all caught: an RMS norm where the L2 norm belongs, key heads
tiled instead of repeated in place, the attention gate dropped, the whole head
rotated instead of a quarter, and the query scaled by the value width. It also
checks that the adapter is merged into the recurrence's projections, which is
where `kev.train` puts it on these bases.

## Done: the state prefix, and what it is worth

Commit "Run the state once, and keep it" — `prefill` / `forward_from` on both
backbones, `Pass::state` and `Pass::branches` to split a request, and an LRU of
prefilled states in `Backend`.

Every question of a request shares the state, so the state runs once and each
question continues from it: an attention layer keeps the state's keys and values,
a recurrent layer keeps its state matrix and the tail of its convolution window.
Exact, because the state comes first and neither layer kind looks forward — which
is the same argument `kev.serve` makes. Across requests, the last four states are
kept, keyed by their token ids, as the server does.

Measured on the synthetic checkpoints (`cargo test --features candle -- --ignored
--nocapture`; tiny models, so read the ratios, not the milliseconds):

| | five questions, 241-token state |
|---|---|
| recurrent, state per question | 57 ms |
| recurrent, state once | 24 ms |
| recurrent, state from the cache | 14 ms |
| attention-only, one packed pass | 20 ms |
| attention-only, prefilled (miss) | 19 ms |

And on a 1200-token state, attention-only: 129 ms packed, 136 ms on a prefix
miss, **20 ms** on a hit.

So the defaults differ, and the reason is measured rather than assumed. A
recurrent base always prefills — it would otherwise run the state per question.
An attention-only base already runs the state once in its packed pass, so a miss
buys nothing and costs a few percent in per-pass overhead, while a hit skips the
state entirely: the prefix path starts at 384 state tokens there, the same
threshold `kev.serve` uses and for the same reason.

`Backend::with_prefix(false)` keeps the packed path, which is what the
transcription tests compare against. Both backbones have a test asserting that
the prefix path answers identically, and the cache has one for hits, misses and
eviction.

## Done: batched branches, and the delta rule in chunks

Commit "Batch the branches, chunk the delta rule".

**Batching.** The branches of one request are padded to the longest and run as one
pass (`forward_from_batch`, `pad_rows`, `branch_batch_mask`). Both backbones took
a batch dimension for it; a pad key is closed to every real query by the mask, a
pad query is left the state to look at so no softmax row is empty, and pads are
never read back. Worth a lot where a pass is short and per-call overhead
dominates (the toy model: branches 11 ms to 4.5 ms), and little where the
arithmetic already fills the CPU (the wide model: 409 to 375 ms).

**Chunks.** `torch_chunk_gated_delta_rule`, transcribed: within a chunk of 64
tokens the updates are condensed into matmuls through a UT transform — pairwise
decays, a unit lower triangular system — and the sequential scan is left one step
per chunk. The inverse of that system is built block by block, recursing on the
two diagonal blocks at once (they are independent, so they go in the batch
dimension): `log2(64)` levels of two matmuls, against `12` matmuls of `64^3` for
the obvious `I + A + A^2 + ...`, which measured 8% of the whole request.

Whether chunking pays depends on shape, and the toy model says the opposite of the
released one, so `chunking_pays` decides per request: more than a chunk of tokens,
and `key * value` above `chunk^2 / 6`. The released checkpoints (128 by 128) are
far above it; the test fixtures far below. `with_chunked_recurrence` forces either
form, which is how the test compares them.

| recurrent base, 128-wide heads, 511 state tokens, 5 questions | |
|---|---|
| token by token | 1930 ms |
| in chunks | 940 ms |
| in chunks, state cached | 375 ms |

`tests/qwen3_5.rs` runs the transcription against both forms — a long row where
the chunked path is exercised properly, and a short one where the padding is
nearly everything. Four deliberate mistakes in the chunked path were caught and
none of them by the short-row test, which is what says the long one earns its
place: no causal mask on the pairwise decays, the inverse truncated, the keys not
decayed to the chunk end, and the strict lower triangle taken inclusive.

## Done: batched prefills, and a chunk size that can be chosen

Commit "Batch the prefills, make the chunk size a knob".

`Forward` gained `hidden_batch`, whose default answers passes one at a time, so a
backend that cannot share anything keeps working.
`LocalEngine::system_one_batch_blocking` takes several requests and hands their
passes over together; `Backend` overrides `hidden_batch` to prefill the distinct
states in one pass and then answer each request's questions as its own batch — the
branches cannot be shared, since every one of them reads its own state.

On a recurrent base, padding a batch of states is not a matter of masking: a
recurrence walks the tokens, and a pad would decay the state and write to it. So
the decay and the write strength are zeroed at pads (`real_mask`), which makes a
short row hand on exactly the state it had at its last real token, and the
convolution window is taken at each row's own end. Removing that masking makes
`answering_several_requests_at_once_gives_the_same_answers` fail, which is how it
is known to matter.

What it is worth, measured:

| eight requests, 20-token states, toy widths | |
|---|---|
| prefilled one by one | 44 ms |
| prefilled together | 32 ms |

| four requests, 200-token states, 128-wide heads | |
|---|---|
| prefilled one by one | 1564 ms |
| prefilled together | 1567 ms |

Which is the expected shape: batching saves per-pass overhead, and a long state
has none worth saving. The first measurement of this said the opposite — batched
prefills 70% *slower* — and the reason was a bug it found: a batch bigger than the
cache evicted the states it had just prefilled and ran them again, one at a time.
The cache now keeps a floor of one batch's worth while the batch is in flight.

The chunk size is `with_chunk_size` now, default 64, a power of two because the
block-by-block triangular inverse halves it down to one. The transcription test
runs at 2, 8 and 64, on a long row and a short one.

## Done: precision, bf16 for serving

Commit "Serve in bf16 where it exists, keep f32 exact".

The backbone runs in whatever `Backend::open_as` is given, and `open_on` picks it
the way `kev.serve` does: **bf16 on a GPU, f32 on the CPU**. What stays f32
whatever the backbone runs in, because the reference keeps it there: the LoRA
merge (before the cast — in bf16 that lands closer to the f32 numbers than merging
after), the softmax, the whole delta rule with its decay and write strength, the
DeltaNet's gated norm and its `A_log`/`dt_bias`, the hidden states handed back,
and the pointer head.

**candle has no bf16 matmul on the CPU** (f16, f32, f64 only), so asking for it
there is refused with a message that says what to use instead, rather than failing
several layers deep in a projection. f16 is the reduced precision a CPU can run,
and it goes through exactly the same code.

Measured on the wide fixture, one request of 536 tokens, no cache:

| | |
|---|---|
| f32 | 707 ms |
| f16 | 567 ms |

and both backbones have a test asserting f16 answers stay within 0.01 of f32.
f32 remains the CPU default anyway: it is the path every published number was
measured at, and this is a model whose whole output is a calibrated probability.

CLAUDE.md and the README describe this now, precision section and all.

## Done: the upstream unit tests, ported

Commit "Port the Python unit tests". `tests/upstream_unit.rs` carries
`tests/test_unit.py` over: the six exact `render` outputs, the record mapping for
all three question types (state text, option texts, keys, legend), the answer
formulas for given distributions, that a rounded 40- and 255-option distribution
still sums to one within TypeSafe's 0.02, the confidence edge cases, the mask rule
with the same indices, and that the temperature divides the logits without moving
the winner. All seven passed first run.

Their worth is that the expectations come from outside this repository. Every
other test here compares the implementation against itself or against a
transcription this repository wrote; these are the reference's own numbers, which
is the closest thing to a parity run available while huggingface.co stays
unreachable from this container (checked again: it is).

The stub backend moved to `tests/fixtures/mod.rs` so both test files share one.

## Done: permute

Commit "Answer permute locally too". `LocalEngine::permute_blocking` and its async
twin run one `choice` question under several option orders and report `runs`,
`argmax_stable` and the per-option `spread` — the same JSON shape
`Client::permute` hands back, so the two backends swap. Raw JSON for the reason
the client's is: the envelope is not in the API docs.

Only the named question is asked, as on the server, and `rounds` is clamped to
1..=64 as the client clamps `n_perm`. The first run keeps the order as given; the
rest are shuffled from the seed with a small generator of our own, which is
reproducible but deliberately *not* CPython's shuffle — the point of the endpoint
is how far the probabilities move, not which permutations were tried. Every run
repeats the same state, so the prefix cache means only the first pays for it.

The engine gained `Error::Invalid`, the local equivalent of the server's 422 for a
request that does not make sense; `is_validation()` covers it.

That closes the API gap: everything the HTTP client offers except `/v1/models`,
which is about a server's own state and has nothing to answer locally.

## Done: option isolation

Commit "Lay options out in isolation when a checkpoint wants it".

The layout an option-isolated checkpoint was trained on: every option span
restarts where the instructions end, all spans share those positions, `<decide>`
sits one past the longest of them, and the mask lets an option be read only by
itself and by `<decide>`. `Pass` carries the per-token `OptionSlot` for it —
`Pass::new` for the ordinary case, `with_options` for this one — and `attends`
gained the extra rule. `Encoding.slots` is empty when isolation is off, so the
usual path carries nothing.

What it buys is asserted over a real backbone in `tests/qwen3.rs`: the same
question with its two options swapped gives each option the same probability. The
same test asserts that the *ordinary* layout does not have that property, so it
cannot pass vacuously. The mask rule itself is ported from
`test_option_isolation_mask_rule`, indices and all.

A recurrent base refuses it, as the Python does: the rule is about who may read
whom, and a recurrence reads everything it walked past.
`option_isolation(head.pt)` reads what a checkpoint was trained with — the pickle
walker needed to learn that Python bools are numbers too — because serving the
wrong layout is a silently different prompt, not an error.

`std::iter::repeat_n` had to go back to `repeat().take()`: it is stable since
1.82 and the crate says 1.75.

## Done: checking it without a server, or Python

Commit "Add a sanity example for a real checkpoint". The question that prompted
it was how to test the engine with no server and no Python at all. Two answers,
in order of what they are worth:

- The whole suite already runs that way: 79 tests with `--features candle`,
  47 with `--features local`, 14 with neither, no weights and no network.
- With a real checkpoint but still no server, `examples/sanity.rs`:

  ```bash
  cargo run --features candle --example sanity -- --base <base> --checkpoint <kev>
  ```

  It asks the two questions that can be answered without reference numbers.
  *Does it hang together* — the same request answered all-questions-at-once
  against one-at-a-time, prefilled-once against per-question, and on a hybrid
  base the delta rule chunked against sequential; those are different code paths
  over the same weights, so the layout, the mask and the recurrence are under
  test even though the comparison is with itself. *Does it mean anything* — seven
  tickets whose answer is not in doubt (a lost parcel is shipping, a double
  charge is billing, "THIRD time writing" escalates, "no rush at all" does not).
  A checkpoint that scores 0.87 held out should take nearly all of them. Non-zero
  exit if a path disagrees or more than one ticket is missed.

Smoke-tested on the synthetic fixtures, where it behaves as designed: all three
consistency checks agree to 0.00000 on the hybrid fixture, and the noise weights
score 2 of 7 with every distribution flat to two decimals — which is what a
broken implementation on real weights would also look like, and is the reason the
plausibility half is there at all.

It also prints the README's worked example next to the numbers published for
Kev-4B (0.47/0.28/0.25, p(yes) 0.93, score 1.44). Those were measured in bf16 on
an M5, so they are a sanity range and not a tolerance — the ordering is the part
worth reading. A checkpoint can be downloaded without Python too: the files are
plain `curl -L https://huggingface.co/<repo>/resolve/main/<file>`.

## Done: a front end, so it can be used and not only tested

Commit "Add a decide example: the server's job in process". Asked what we had
built and how to *use* kev and qwen without Python or a server, the honest answer
was that every example so far either needed the server (`triage`) or was a check
(`parity`, `sanity`) — the library was usable, the repo had no tool.
`examples/decide.rs` is that tool:

```bash
cargo run --release --features candle --example decide -- \
    --base <base> --checkpoint <kev> --state "..."          # or --request r.json
cat tickets.txt | cargo run --release --features candle --example decide -- \
    --base <base> --checkpoint <kev> --questions q.json --lines --json
```

- `--request` repeats into a batch (one prefill pass over the states),
  `--state`/`--questions` covers the ad-hoc case, and `--lines` answers a state
  per line with the model loaded once and the prefix cache at 16 — verified that
  a `--lines` run gives the same answers as separate single runs, distinct states
  included, which is the property that makes the cache safe to leave on.
- `--json` prints the server's envelope around `answers_json`, which had to
  become public for it (it was `pub(crate)` for `usage.output_tokens`). The
  answers are byte-compatible with the Python; the envelope is assembled in the
  example, and says so.
- `--separate`, `--permute`, `--rounds`/`--seed`, `--dtype`, and an option-layout
  override. Diagnostics on stderr, answers on stdout, so `--json | jq` works.
- Smoke-tested on both synthetic fixtures, every mode: the batched and the
  separate path agree on the probabilities and differ on `input_tokens` (36 vs
  41, the state counted once per question), which is exactly the documented
  behaviour.

On the way: `temperature()` and `option_isolation()` answered "that is not a
torch archive" for a head exported as safetensors, which `pointer_head` reads
happily. They now answer `None`, decided by the extension, with a test.

## Done: one script for the whole server-free check

Commit "Add scripts/local.sh". `scripts/local.sh` is the sequence, in the order
worth running it, and the answer to "a script for testing locally, no Python, no
server":

```bash
scripts/local.sh                                  # the offline suite alone
scripts/local.sh --fetch jaredpalmer/kev-0.6b     # download, then everything
scripts/local.sh --checkpoint <dir> [--measure] [--skip-suite] [--base <dir>]
```

Steps: `scripts/test.sh`, then `examples/sanity`, then the suite again with
`KEV_TOKENIZER` pointed at the checkpoint's own `tokenizer.json`, then one request
through `examples/decide`, and with `--measure` the `#[ignore]`d timings. Failures
are collected and summarised rather than stopping the run, as `test.sh` does.

Decisions worth keeping:

- **`--no-default-features --features candle` throughout.** The first run pulled
  in reqwest and therefore `ring`, which needs a C compiler this container does
  not have — and no step here talks HTTP anyway.
- **`curl`, not `hf`.** The CLI is a Python package, so it is out by definition.
  The script takes the adapter, `head.pt` and the tokenizer, reads
  `base_model_name_or_path` out of `adapter_config.json` (stripping a `@rev`
  suffix), and fetches that base — including sharded weights, whose names it
  takes out of `model.safetensors.index.json`. `--force` re-downloads, and also
  deletes first: `-C -` against a complete file is a range request past the end,
  which the hub answers with 416.
- **A merged checkpoint passes no `--checkpoint`.** A directory with a
  `config.json` and no `adapter_config.json` *is* the base, and asking candle for
  an adapter that is not there is an error, not a no-op. That is also what the
  test fixtures look like, which is how it surfaced.
- **The head is checked before anything is built.** No `head.pt` and there is
  nothing to read answers off; saying so after a release build of candle would be
  unkind.
- **`KEV_HF`** overrides the hub URL. It is there for a mirror, and it is how the
  download path was tested here: a `file://` tree with a fake checkpoint, a
  sharded fake base, and missing optional files, which exercised every branch
  except the transport.

Verified on both synthetic fixtures (attention-only and hybrid): all three steps
run, `sanity` reports 2 of 7 obvious cases and 0.00000 across the paths as it
should on noise weights, and the script's exit code follows it. The error paths
were run too: no head, no such directory, unknown argument, `--skip-suite` with
no checkpoint.

## Done: a real checkpoint, for the first time

Run by the user on their Mac on 2026-09-23 (`scripts/local.sh --fetch
jaredpalmer/kev-0.6b`, cargo 1.97.1, f32 on an Apple CPU), which is the first
time this code has ever seen real weights — the container cannot reach the hub.
What came back:

- **7 of 7 obvious cases** in `examples/sanity.rs`: `shipping` 1.00 for a lost
  parcel, `returns` 1.00 for a wrong size, `billing` 0.94 for a double charge,
  `p(yes)` 0.71 on "THIRD time writing" against 0.01 on a thank-you note,
  frustration 1.94 and urgency 0.08. On noise weights the same run scores 2 of 7
  with every distribution flat at 0.33, so this is the discriminating result:
  a wrong rotary convention, token layout, LoRA merge or head would not
  saturate.
- `head.pt says option_isolation = false` — the real torch pickle parsed,
  projections, temperature and metadata included, and the released checkpoints do
  leave isolation off as the Python says.
- `hidden_size: 1024` with the head agreeing: weight names and shapes are right.
- `largest difference 0.00000` for packed against separate and for prefilled
  against per-question, now over a real layer stack rather than a two-layer toy.
- The suite passed again with `KEV_TOKENIZER` set to the checkpoint's own
  `tokenizer.json`, `a_real_qwen_tokenizer_carries_the_five_delimiters` included.
- One request through `decide`: 101 tokens in, 1671 ms, model loaded in 1.0 s.

The one difference is the README's worked example: this checkpoint answers
`billing` 0.51 (shipping 0.25, returns 0.24), `escalate` 0.41, frustration 1.04,
where the README publishes `returns` 0.47, 0.93 and 1.44. That is **Kev-4B in
bf16 against Kev-0.6B in f32** on a ticket that deliberately names all three
departments at once, so a smaller model disagreeing there is expected and is not
evidence of a bug — which is exactly why `sanity` judges on unambiguous cases and
only prints the 4B numbers beside its own.

What this does *not* establish is parity: whether the Python answers `kev-0.6b`
with these same numbers. That still needs the server once.

## Done: the measurements, on a real checkpoint

Commit "Add a measure example". `examples/measure.rs` asks the performance
questions of real weights instead of the fixtures, and `scripts/local.sh
--measure` now runs it before the `#[ignore]`d fixture measurements:

```bash
cargo run --release --features candle --example measure -- \
    --base <base> --checkpoint <kev> [--words 400] [--repeat 3] [--batch 4]
```

Rows: the packed path, the prefix with nothing kept, the prefix with a cache hit,
f16 against f32 in time *and* in largest answer difference, the delta rule
sequential against chunked on a hybrid base, and several requests prefilled
together against one at a time. Then the ratios, which are the only part worth
quoting.

Four decisions, all of them about not fooling ourselves — the bogus f16/f32
comparison earlier in this task is the reason:

- **`with_prefix_min_tokens(0)`.** An attention-only base skips the prefix below
  384 state tokens, so without this both prefix rows would measure the packed path
  and agree beautifully.
- **A control row with the cache off.** The cached row is only evidence if it sits
  far below the same configuration keeping nothing.
- **f16's largest difference from f32, in the same run.** A speed-up that moves
  the probabilities is not a speed-up, and quoting a difference measured elsewhere
  is how that gets missed.
- **A failed pass is an error, not a fast one.** The first draft of `times()`
  discarded the result, which would have printed an excellent number for a
  request that never answered.

Each cold pass gets its own state, so it is a real miss rather than the same state
twice; each row is a median with min and max printed beside it.

Verified here on both synthetic fixtures — the harness runs, and the numbers
reproduce what the fixture measurements already said: at toy widths the prefix
pays on a recurrent base (2.2x) and not on an attention-only one, chunking does
not pay (0.87x), f16 is roughly neutral. **The numbers that matter are the ones
from the real checkpoint, and only the user's Mac can produce them** — with a
400-word state the container's hybrid fixture shows the shape (cache 3.6x, f16
1.15x) but not the magnitudes.

## Done: the HTTP client is gone

Commit "Remove the HTTP client and make the model the default feature". The user
does not need the path through a running Python server any more, so it is out
rather than carried along: dead weight in a crate is worse than a missing feature,
and `reqwest` was the crate's only dependency that needed a C toolchain.

Removed: `src/client.rs`, `examples/triage.rs`, `scripts/triage.sh`, the `http`
feature and `reqwest`, `tests/seam.rs`'s `over_http` module. With them went the
error variants that only existed for HTTP — `InvalidBaseUrl`, `Transport`, `Api`,
`Decode` — and `SystemOneResponse.request_id`, which came from a response header.

Kept, and why:

- **The `SystemOne` trait.** It was designed as the seam between backends and it
  still is one: a recording, a queue, another engine. It needs no feature, and
  `tests/seam.rs` still proves a backend with no model in it can implement it with
  `--no-default-features`.
- **The `Send` bound tests.** They used to be about `Client`; the property is about
  the trait, so they moved to `LocalEngine` (three tests in `over_the_local_engine`,
  using the stub from `tests/fixtures`). `#[path]` resolves relative to the module
  it sits in, so the fixtures module has to be declared at the test crate's root.
- **`Error::Invalid` is now unconditional.** With every variant behind a feature, a
  `--no-default-features` build would have an *uninhabited* `Error`, and `Display`
  cannot exhaustively match a reference to one. One always-present variant, and the
  question does not arise.

Two deliberate consequences:

- **`default = ["candle"]`.** The model is what the crate is for, and with `http`
  gone the old default named a feature that no longer existed. Every documented
  command lost its `--no-default-features --features candle`, and `cargo test`,
  `cargo run --example decide` now work as they read.
- **The whole feature matrix builds in this container for the first time.**
  `scripts/test.sh` ran green here end to end — fmt, clippy `--all-features`,
  tests, matrix — because nothing left needs `cc`. Previously `--all-features`
  could not even be attempted.

Tests: 83 with the default features (was 80, plus the three moved seam tests), 50
with `local`, 14 with none. `scripts/test.sh` green.

## Left to do

1. **Parity — the only thing left that needs the server.** The loading, the real
   vocabulary and the real `head.pt` are now checked (above): `kev-0.6b` loads and
   answers sensibly, so what is unverified is narrower than it was — whether the
   Python gives the *same* numbers for the same request. `examples/parity.rs`
   answers a recorded request in process and compares every probability with a
   recorded server response (`--tolerance`, non-zero exit past it). What is left is
   starting `kev.serve` once with `KEV_DTYPE=fp32` on the same checkpoint the local
   run used, recording request and response, and turning that pair into an offline
   fixture so it never needs starting again. Expect small differences from the
   server's own dtype, and read the argmax and the ordering before the decimals.
2. **Performance, what is left of it.** The prefix, the batched branches, the
   chunked delta rule, the batched prefills and the precision knob are in
   (below). Still open: nothing is quantised, no device other than the CPU has
   been run at all, and a *server* would want to collect concurrent requests into
   a batch itself — `system_one_batch_blocking` is the call it would make, not the
   queue in front of it.
3. Nothing else from the original plan. What is left is the parity run, and
   whatever a real checkpoint turns out to need.

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
    cargo test --no-default-features --features candle
```

`+fp16` is for `gemm-f16`, whose inline assembly does not build on this
aarch64 target without it; a Mac has it in the base feature set.

Verified this way on 2026-09-23 (cargo 1.98.1): `--no-default-features`,
`--features local` and `--features candle` all green — 6 backbone tests, 15
engine tests, 2 seam tests, 10 wire-format tests, 1 doctest — with
`cargo fmt --check` and `cargo clippy --all-targets` clean.

The `http` feature still cannot be built here: `ring`'s build script wants a
real C compiler. Anything HTTP-side stays verified only on the Mac.
