# Parity requests

The requests `scripts/parity.sh` sends to a Python Kev server and then answers
locally. Each one is here because it can break in a way the offline tests cannot
see — those compare the implementation with itself or with a transcription, never
with the reference running.

| Request | What it is there to catch |
|---|---|
| `triage.json` | the worked example from the Kev README: `choice`, `noul` and `score` in one request, with published numbers to sanity-check against |
| `structured-state.json` | a JSON state rather than text, so `render` has to reproduce Python's `str()` — `True`, `None`, floats, nested objects and a list of messages |
| `many-options.json` | twelve options, two of them with `null` descriptions: option slots, the `</opt>` readout at scale, and 4-decimal rounding that still has to sum to one |
| `escaping.json` | text that *contains* Kev's delimiters (`<\|fim_prefix\|>`, `</opt>`, `<decide>`) plus an injected `<\|im_start\|>` instruction, umlauts, CJK and emoji: the escaping, and the `\uXXXX` in `usage.output_tokens` |
| `long-state.json` | over 400 words, past the 384-token threshold, so the prefix path answers rather than the packed one |

Every one is sent to `/v1/systemone` **and** `/v1/systemone/separate`, so both
paths are compared; the recordings land in `recordings/` beside this file.

Keep the recordings once they exist. `scripts/parity.sh --check-only` then repeats
the whole comparison from the recordings alone, which is what turns a one-off
session at a server into a standing check.

The tolerance is about the server's precision, not about being lenient: record
with `KEV_DTYPE=fp32` and the differences should be at the fourth decimal. A
difference in `input_tokens` is not a tolerance question at all — it means the two
sides tokenised different prompts, and the script fails on it whatever the
probabilities say.
