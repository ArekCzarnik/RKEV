#!/usr/bin/env bash
#
# Measure one checkpoint through every path it has: CPU and Metal, f32 and f16,
# dense and quantised — and what each costs the answers.
#
#   scripts/sweep.sh --base ~/models/qwen-qwen3-0.6b-base --checkpoint ~/models/kev-0.6b
#   scripts/sweep.sh --base <dir> --checkpoint <dir> --skip-metal
#   scripts/sweep.sh --base <dir> --checkpoint <dir> --quantise "q8_0 q4k" --repeat 7
#
#   --repeat <n>      passes per row, median reported (default 5)
#   --words <n>       length of the generated state (default 400, over the
#                     384-token prefix threshold on purpose)
#   --quantise <list> which stages to try (default "q8_0 q6k q4k"); "" for none
#   --skip-metal      CPU only
#   --skip-cpu        Metal only
#   --out <file>      where the log goes (default rkev-measurements.txt)
#
# Nothing here is fatal: a run that fails is noted and the sweep carries on, since
# a missing Metal op should not cost you the CPU numbers. The whole log is written
# to --out as well as to the terminal, and the ratio lines are collected at the end
# — that summary is what is worth sending on.
#
# The first Metal build recompiles candle's backend and takes minutes.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CRATE_DIR="$ROOT/rkev"

base=""
checkpoint=""
repeat=5
words=400
stages="q8_0 q6k q4k"
skip_metal=0
skip_cpu=0
out="rkev-measurements.txt"

while [ $# -gt 0 ]; do
    case "$1" in
        --base)       base="${2:-}"; shift 2 ;;
        --checkpoint) checkpoint="${2:-}"; shift 2 ;;
        --repeat)     repeat="${2:-}"; shift 2 ;;
        --words)      words="${2:-}"; shift 2 ;;
        --quantise|--quantize) stages="${2:-}"; shift 2 ;;
        --skip-metal) skip_metal=1; shift ;;
        --skip-cpu)   skip_cpu=1; shift ;;
        --out)        out="${2:-}"; shift 2 ;;
        -h|--help)    sed -n '3,24p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)            echo "error: unknown argument $1 (try --help)" >&2; exit 1 ;;
    esac
done

[ -n "$base" ] || { echo "error: --base is required" >&2; exit 1; }
[ -d "$base" ] || { echo "error: no directory $base" >&2; exit 1; }
if [ -n "$checkpoint" ] && [ ! -d "$checkpoint" ]; then
    echo "error: no directory $checkpoint" >&2
    exit 1
fi

# Absolute, because the runs happen from inside the crate directory.
case "$base" in /*) ;; *) base="$PWD/$base" ;; esac
case "$out"  in /*) ;; *) out="$PWD/$out" ;; esac
if [ -n "$checkpoint" ]; then
    case "$checkpoint" in /*) ;; *) checkpoint="$PWD/$checkpoint" ;; esac
fi

# rustup installs into ~/.cargo but only a login shell picks that up.
if ! command -v cargo >/dev/null 2>&1 && [ -f "${CARGO_HOME:-$HOME/.cargo}/env" ]; then
    # shellcheck disable=SC1091
    . "${CARGO_HOME:-$HOME/.cargo}/env"
fi
command -v cargo >/dev/null 2>&1 || {
    echo "error: cargo is not on PATH - no Rust toolchain here." >&2
    exit 127
}

# What is in those directories, checked before anything is compiled. Ten runs that
# all die on the same missing file are ten wasted builds and a summary that reads
# like a finding.
lacks() {
    echo "error: $1" >&2
    shift
    while [ $# -gt 0 ]; do
        echo "       $1" >&2
        shift
    done
    exit 1
}
contents() {
    local listing
    listing="$(ls -1 "$1" 2>/dev/null | head -6 | tr '\n' ' ')"
    echo "it holds: ${listing:-<nothing>}"
}

[ -f "$base/config.json" ] || lacks \
    "no config.json in $base" \
    "--base is the base model: config.json and its safetensors." \
    "$(contents "$base")"
ls "$base"/*.safetensors >/dev/null 2>&1 || lacks \
    "no safetensors in $base" \
    "A sharded base model has model-00001-of-*.safetensors and an index." \
    "$(contents "$base")"

if [ -n "$checkpoint" ]; then
    [ -f "$checkpoint/adapter_config.json" ] || lacks \
        "no adapter_config.json in $checkpoint" \
        "--checkpoint is the Kev checkpoint: adapter_config.json," \
        "adapter_model.safetensors, head.pt, and usually a tokenizer." \
        "$(contents "$checkpoint")" \
        "" \
        "If the weights are already merged into --base, leave --checkpoint out." \
        "If it was never downloaded: scripts/local.sh --fetch jaredpalmer/kev-0.6b"
    [ -f "$checkpoint/head.pt" ] || [ -f "$checkpoint/head.safetensors" ] || lacks \
        "no head.pt in $checkpoint" \
        "A Kev checkpoint is an adapter plus a pointer head, and the answers" \
        "come out of the head." \
        "$(contents "$checkpoint")"
fi

model=(--base "$base")
[ -n "$checkpoint" ] && model+=(--checkpoint "$checkpoint")

cd "$CRATE_DIR"
: > "$out"
failed=""
said_no=""

# Every run: announced, timed by the example itself, and kept in the log.
run() {
    local label="$1"
    shift
    {
        echo ""
        echo "########## $label"
        echo "\$ $*"
    } | tee -a "$out"
    if "$@" 2>&1 | tee -a "$out"; then
        return 0
    fi
    failed="${failed}${failed:+ | }${label}"
    echo "!!!!!!!!!! $label did not finish" | tee -a "$out"
    return 1
}

# The same, for a step whose non-zero exit is a verdict rather than a fault:
# sanity says no when a checkpoint misses the obvious cases, and that is the
# information, not a broken run.
run_verdict() {
    local label="$1"
    shift
    {
        echo ""
        echo "########## $label"
        echo "\$ $*"
    } | tee -a "$out"
    local step
    step="$(mktemp)"
    if "$@" > "$step" 2>&1; then
        cat "$step" | tee -a "$out"
        rm -f "$step"
        return 0
    fi
    cat "$step" | tee -a "$out"
    # A verdict has a tally in it. Without one the run never got far enough to
    # judge anything - a checkpoint that would not load is a failure, not a no.
    if grep -q "obvious cases" "$step"; then
        said_no="${said_no}${said_no:+ | }${label}"
    else
        failed="${failed}${failed:+ | }${label}"
        echo "!!!!!!!!!! $label did not finish" | tee -a "$out"
    fi
    rm -f "$step"
}

{
    echo "rkev measurement sweep"
    echo "date      $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "machine   $(uname -srm)"
    echo "cargo     $(cargo --version)"
    echo "base      $base"
    echo "checkpoint ${checkpoint:-<none, the base carries its own weights>}"
    echo "settings  repeat=$repeat words=$words quantise=\"$stages\""
} | tee -a "$out"

# ---------------------------------------------------------------------------
# What to read, and what to send on
# ---------------------------------------------------------------------------

summarise() {
{
    echo ""
    echo "########## the lines worth quoting"
    grep -E "^(#########|attention-only base|hybrid base|.*rather than f32|.*against f32, largest difference|the cache saves|running the state once|the delta rule in chunks|[0-9]+ of [0-9]+ obvious cases|[0-9]+ of [0-9]+ with the head)" "$out" \
        | sed 's/^##########/--/'
    echo ""
    if [ -n "$failed" ]; then
        echo "did not finish: $failed"
        case "$failed" in
            *q4k*|*q5k*|*q6k*|*q8_0*)
                echo "For a quantised run, the refusal is often the answer: a projection whose"
                echo "row does not divide by the block size says so by name, and q8_0 packs 32"
                echo "where the k-quants pack 256."
                ;;
        esac
    else
        echo "every run finished."
    fi
    if [ -n "$said_no" ]; then
        echo ""
        echo "not convinced: $said_no"
        echo "Those runs finished and said no: the checkpoint missed more than one of the"
        echo "obvious cases. On made-up weights that is expected; on a real checkpoint it"
        echo "is the finding, and reading it beats reading the timings above it."
    fi
    echo ""
    echo "the whole log: $out"
} | tee -a "$out.summary"

cat "$out.summary" >> "$out"
rm -f "$out.summary"
}
# ---------------------------------------------------------------------------
# CPU
# ---------------------------------------------------------------------------

if [ "$skip_cpu" -eq 0 ]; then
    # If the baseline cannot run, nothing below it can either, and repeating the
    # same failure nine times buries the one line that matters.
    if ! run "cpu, dense" \
        cargo run --release --example measure -- \
        "${model[@]}" --repeat "$repeat" --words "$words"; then
        echo "" | tee -a "$out"
        echo "stopping: the plain CPU run did not get through, so the rest would fail" \
            "the same way. The line above it says why." | tee -a "$out"
        summarise
        exit 1
    fi

    for stage in $stages; do
        run "cpu, $stage" \
            cargo run --release --example measure -- \
            "${model[@]}" --repeat "$repeat" --words "$words" --quantise "$stage"
    done

    # What the speed costs the decisions. The dense run is the reference; if it
    # does not take the obvious cases, nothing below it means anything.
    run_verdict "cpu, dense: the obvious cases" \
        cargo run --release --example sanity -- "${model[@]}"

    for stage in $stages; do
        run_verdict "cpu, $stage: the obvious cases" \
            cargo run --release --example sanity -- "${model[@]}" --quantise "$stage"
    done
fi

# ---------------------------------------------------------------------------
# Metal
# ---------------------------------------------------------------------------

if [ "$skip_metal" -eq 0 ]; then
    if [ "$(uname -s)" != "Darwin" ]; then
        echo "" | tee -a "$out"
        echo "skipping Metal: this is $(uname -s), and candle's Metal backend is macOS only" \
            | tee -a "$out"
    else
        echo "" | tee -a "$out"
        echo "the first Metal run rebuilds candle's backend; give it a few minutes" \
            | tee -a "$out"
        run "metal, dense" \
            cargo run --release --features metal --example measure -- \
            "${model[@]}" --repeat "$repeat" --words "$words" --device metal
        run_verdict "metal, dense: the obvious cases" \
            cargo run --release --features metal --example sanity -- \
            "${model[@]}" --device metal
        # Quantisation is a CPU path here, so there is deliberately no metal row
        # for it - asking would be refused, and the refusal is the documentation.
    fi
fi


summarise
[ -z "$failed" ]
