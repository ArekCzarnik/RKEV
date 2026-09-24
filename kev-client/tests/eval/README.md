# Eval records

What `examples/eval.rs` reads: a questions file, and one record per line with the
answers you consider right. `tickets.jsonl` and `questions.json` here are a
six-record sample, in German, to copy the shape from — not a benchmark.

```bash
cargo run --release --example eval -- \
    --base <base> --checkpoint <kev> \
    --questions tests/eval/questions.json --records tests/eval/tickets.jsonl
```

**`--questions`** is the `questions` map of a System One request, exactly as you
would send it. `--request <file>` takes the map out of a whole request instead, so
a file you already use for `decide` or `parity` works.

**`--records`** is JSONL, one object per line:

```json
{"state": "…the ticket…", "labels": {"abteilung": "versand", "eskalation": true, "verärgerung": 1}}
```

- `state` is a string or any JSON object or array, the same as in a request.
- `labels` holds the answer you consider right, keyed by question id:
  - a **choice** takes the option name (`"versand"`),
  - a **noul** takes `true` or `false`,
  - a **score** takes the level index (`0` for the first level) or its description
    (`"ruhig"`).
- A question you leave out of `labels` is simply not scored for that record, so a
  partly labelled set is fine. A label for a question that does not exist is an
  error rather than a silent zero — a typo in a question id would otherwise look
  like a perfect score on nothing.

Every question is answered for every record either way: the model never sees the
labels, and the questions cannot read each other.

## What it reports

Per question, the measure that fits its type: accuracy for a choice (with the
confusion between options where they fit), accuracy at 0.5 plus the separation
between the two classes and the AUC for a noul, and nearest-level accuracy plus the
mean absolute error for a score.

Then the part a decision model is actually for: **accuracy by confidence**. A
calibrated model is one you can route on automatically above some confidence and
send to a person below it, and that table is where that threshold comes from.
`--errors <n>` prints the confident mistakes, which is where a question's wording
usually turns out to be the problem.
