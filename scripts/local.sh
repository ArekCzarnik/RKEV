#!/usr/bin/env bash
#
# Check the local inference engine without a Kev server and without Python.
#
#   scripts/local.sh                            # the offline suite; needs no weights
#   scripts/local.sh --checkpoint ~/models/kev-0.6b
#                                               # ... plus every check a checkpoint allows
#   scripts/local.sh --fetch jaredpalmer/kev-0.6b
#                                               # download it first, with curl
#   scripts/local.sh --checkpoint <dir> --measure   # also the timing measurements
#   scripts/local.sh --checkpoint <dir> --skip-suite  # only the checkpoint checks
#
#   --base <dir>      the base model, if it is not beside the checkpoint
#   --dir <dir>       where --fetch puts models (default ~/models)
#   --force           re-download files (also the fix for an interrupted one)
#   --release         build the suite optimised too (sanity always is)
#
# Nothing here starts a server, imports Python, or needs the network unless
# --fetch is given. HF_TOKEN is used for --fetch if it is set.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CRATE_DIR="$ROOT/kev-client"
MODEL_DIR="${KEV_MODEL_DIR:-$HOME/models}"
# KEV_HF points the downloads elsewhere: a mirror, or a local tree (file://...)
# to exercise this script without the network.
HF="${KEV_HF:-https://huggingface.co}"

base=""
checkpoint=""
fetch=""
measure=0
skip_suite=0
force=0
release=""

while [ $# -gt 0 ]; do
    case "$1" in
        --checkpoint) checkpoint="${2:-}"; shift 2 ;;
        --base)       base="${2:-}"; shift 2 ;;
        --fetch)      fetch="${2:-}"; shift 2 ;;
        --dir)        MODEL_DIR="${2:-}"; shift 2 ;;
        --measure)    measure=1; shift ;;
        --skip-suite) skip_suite=1; shift ;;
        --force)      force=1; shift ;;
        --release)    release="--release"; shift ;;
        -h|--help)    sed -n '3,19p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)            echo "error: unknown argument $1 (try --help)" >&2; exit 1 ;;
    esac
done

# rustup installs into ~/.cargo but only a login shell picks that up.
if ! command -v cargo >/dev/null 2>&1 && [ -f "${CARGO_HOME:-$HOME/.cargo}/env" ]; then
    # shellcheck disable=SC1091
    . "${CARGO_HOME:-$HOME/.cargo}/env"
fi
if ! command -v cargo >/dev/null 2>&1; then
    echo "error: cargo is not on PATH - no Rust toolchain in this environment." >&2
    echo "       install one with:  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh" >&2
    exit 127
fi

# Every step runs even when an earlier one failed, and the failures are
# summarised at the end, as scripts/test.sh does it.
failed=""
step() {
    local label="$1"
    shift
    echo ""
    echo "==> $label"
    if ! "$@"; then
        failed="${failed}${failed:+, }${label}"
    fi
}

# ---------------------------------------------------------------------------
# Fetching a checkpoint, without Python: the `hf` CLI is itself a Python
# package, and every file on the hub is a plain HTTPS GET.
# ---------------------------------------------------------------------------

auth=()
[ -n "${HF_TOKEN:-}" ] && auth=(-H "Authorization: Bearer $HF_TOKEN")

# One file off the hub. -C - resumes a half-downloaded shard, and --fail turns a
# 404 into an exit code instead of writing the error page into the file.
fetch_one() {
    curl -fL --retry 3 --retry-delay 2 -C - --progress-bar \
        ${auth[@]+"${auth[@]}"} -o "$3" "$HF/$1/resolve/main/$2"
}

# get <repo> <file> <into> [optional]

# Returns 1 for a missing file unless told it is optional, so a repo that
# simply does not carry one (no generation_config.json, say) is not an error.
get() {
    local repo="$1" file="$2" into="$3" optional="${4:-}"
    local target="$into/$file"
    if [ "$force" -eq 0 ] && [ -s "$target" ]; then
        # Present and non-empty is taken as done: the hub's size is not known
        # here, so an interrupted download looks the same. --force is the cure.
        echo "    have $file"
        return 0
    fi
    # A complete file plus -C - is a range request past the end, which the hub
    # answers with 416; start over instead.
    [ "$force" -eq 1 ] && rm -f "$target"
    mkdir -p "$(dirname "$target")"
    # An optional file the repo does not carry is not news, and neither is
    # curl's complaint about it.
    if [ -n "$optional" ]; then
        fetch_one "$repo" "$file" "$target" 2>/dev/null && { echo "    got  $file"; return 0; }
    else
        fetch_one "$repo" "$file" "$target" && { echo "    got  $file"; return 0; }
    fi
    rm -f "$target"
    if [ -n "$optional" ]; then
        echo "    (no $file in $repo)"
        return 0
    fi
    echo "error: $repo has no $file, or the hub is unreachable" >&2
    return 1
}

# The weights are one file in small repos and sharded in large ones, and the
# index names the shards.
get_weights() {
    local repo="$1" into="$2"
    if [ "$force" -eq 0 ] && ls "$into"/*.safetensors >/dev/null 2>&1; then
        echo "    have safetensors"
        return 0
    fi
    if get "$repo" model.safetensors "$into" optional && [ -s "$into/model.safetensors" ]; then
        return 0
    fi
    get "$repo" model.safetensors.index.json "$into" || return 1
    # The index is JSON, but jq is not everywhere; the shard names are literal.
    local shard
    for shard in $(grep -o 'model-[0-9]*-of-[0-9]*\.safetensors' \
                   "$into/model.safetensors.index.json" | sort -u); do
        get "$repo" "$shard" "$into" || return 1
    done
}

if [ -n "$fetch" ]; then
    if ! command -v curl >/dev/null 2>&1; then
        echo "error: --fetch needs curl" >&2
        exit 1
    fi
    checkpoint_dir="$MODEL_DIR/$(basename "$fetch")"
    echo "==> fetching $fetch into $checkpoint_dir"
    mkdir -p "$checkpoint_dir"
    for file in adapter_config.json adapter_model.safetensors head.pt; do
        get "$fetch" "$file" "$checkpoint_dir" || exit 1
    done
    for file in tokenizer.json tokenizer_config.json README.md; do
        get "$fetch" "$file" "$checkpoint_dir" optional || exit 1
    done
    checkpoint="$checkpoint_dir"

    # Which base it adapts is in adapter_config.json; fetch that too.
    repo="$(sed -n 's/.*"base_model_name_or_path"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' \
            "$checkpoint_dir/adapter_config.json" | head -1)"
    # A checkpoint may name a base with a revision suffix (kev's @qwen3 style).
    repo="${repo%%@*}"
    if [ -z "$repo" ]; then
        echo "error: $checkpoint_dir/adapter_config.json names no base model" >&2
        exit 1
    fi
    if [ -z "$base" ]; then
        base="$MODEL_DIR/$(echo "$repo" | tr 'A-Z/' 'a-z-')"
    fi
    echo "==> fetching base $repo into $base"
    mkdir -p "$base"
    for file in config.json tokenizer.json; do
        get "$repo" "$file" "$base" || exit 1
    done
    get "$repo" generation_config.json "$base" optional || exit 1
    get_weights "$repo" "$base" || exit 1
fi

# ---------------------------------------------------------------------------
# The offline suite: no weights, no network, no server
# ---------------------------------------------------------------------------

if [ "$skip_suite" -eq 0 ]; then
    step "the offline suite (fmt, clippy, tests, feature matrix)" \
        "$ROOT/scripts/test.sh" ${release:+"$release"}
fi

# ---------------------------------------------------------------------------
# What a real checkpoint adds
# ---------------------------------------------------------------------------

if [ -z "$checkpoint" ]; then
    if [ -n "$failed" ]; then
        echo ""
        echo "==> FAILED: $failed" >&2
        exit 1
    fi
    echo ""
    if [ "$skip_suite" -eq 0 ]; then
        echo "==> the offline suite passed. It runs on made-up weights, so it says"
        echo "    nothing about loading a real checkpoint - pass --checkpoint <dir>,"
        echo "    or --fetch jaredpalmer/kev-0.6b to download one first."
    else
        echo "==> nothing ran: --skip-suite leaves only the checkpoint checks, and"
        echo "    there is no --checkpoint." >&2
        exit 1
    fi
    exit 0
fi

[ -d "$checkpoint" ] || { echo "error: no directory $checkpoint" >&2; exit 1; }

# A base beside the checkpoint, or named by its adapter_config.json.
if [ -z "$base" ]; then
    if [ -f "$checkpoint/config.json" ] && ls "$checkpoint"/*.safetensors >/dev/null 2>&1 \
       && [ ! -f "$checkpoint/adapter_config.json" ]; then
        base="$checkpoint"   # an already-merged checkpoint carries its own base
    elif [ -f "$checkpoint/adapter_config.json" ]; then
        repo="$(sed -n 's/.*"base_model_name_or_path"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' \
                "$checkpoint/adapter_config.json" | head -1)"
        repo="${repo%%@*}"
        guess="$MODEL_DIR/$(echo "$repo" | tr 'A-Z/' 'a-z-')"
        if [ -d "$guess" ]; then
            base="$guess"
        else
            echo "error: $checkpoint adapts $repo, which is not at $guess." >&2
            echo "       pass --base <dir>, or --fetch to download both." >&2
            exit 1
        fi
    else
        echo "error: $checkpoint has neither a config.json nor an adapter_config.json" >&2
        exit 1
    fi
fi

[ -f "$base/config.json" ] || { echo "error: no $base/config.json" >&2; exit 1; }
ls "$base"/*.safetensors >/dev/null 2>&1 || {
    echo "error: no safetensors in $base" >&2; exit 1; }

tokenizer="$checkpoint/tokenizer.json"
[ -f "$tokenizer" ] || tokenizer="$base/tokenizer.json"
[ -f "$tokenizer" ] || { echo "error: no tokenizer.json in $checkpoint or $base" >&2; exit 1; }

# A checkpoint with no adapter is already merged into its base, and asking for
# the adapter anyway is an error rather than a no-op.
model=(--base "$base")
if [ -f "$checkpoint/adapter_config.json" ]; then
    model+=(--checkpoint "$checkpoint")
fi

# The pointer head is the other half of a checkpoint; without it there is
# nothing to read the answers off, and that is worth saying before a build.
head_file=""
for candidate in "$checkpoint/head.pt" "$checkpoint/head.safetensors" \
                 "$base/head.pt" "$base/head.safetensors"; do
    [ -f "$candidate" ] && { head_file="$candidate"; break; }
done
if [ -z "$head_file" ]; then
    echo "error: no head.pt in $checkpoint - a Kev checkpoint is an adapter plus a" >&2
    echo "       pointer head, and the answers come out of the head." >&2
    exit 1
fi

cd "$CRATE_DIR"
echo ""
echo "==> base       $base"
echo "==> checkpoint $checkpoint"
echo "==> head       $head_file"
echo "==> tokenizer  $tokenizer"

# Always --release: a real checkpoint under a debug build is minutes per pass.
# --no-default-features throughout: none of this talks HTTP, and reqwest drags
# in ring, which needs a C compiler this crate otherwise does not.
step "the engine against itself and against obvious cases (example sanity)" \
    cargo run --release --no-default-features --features candle --example sanity -- \
    "${model[@]}"

# KEV_TOKENIZER turns on the checks that need a real Qwen vocabulary: the
# specials Kev reuses as delimiters have to be single known ids, and text that
# looks like one must not become one.
step "the suite again with the real tokenizer (KEV_TOKENIZER)" \
    env KEV_TOKENIZER="$tokenizer" cargo test --release --no-default-features --features candle

# One answer through the front end, which is the thing you would actually run.
step "one request end to end (example decide)" \
    cargo run --release --no-default-features --features candle --example decide -- \
    "${model[@]}" \
    --state "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card."

if [ "$measure" -eq 1 ]; then
    # On the real checkpoint: what f16 buys, and what the prefix cache buys, with
    # the controls that keep a cache hit from being read as a precision win.
    step "the timings on this checkpoint (example measure)" \
        cargo run --release --no-default-features --features candle --example measure -- \
        "${model[@]}"
    # And on the synthetic fixtures, which is where the released checkpoints'
    # layer widths are reproduced without their weights.
    step "the timings on the fixtures (#[ignore]d tests)" \
        cargo test --release --no-default-features --features candle -- --ignored --nocapture
fi

if [ -n "$failed" ]; then
    echo ""
    echo "==> FAILED: $failed" >&2
    exit 1
fi

echo ""
echo "==> everything that can be checked without a server passed."
echo "    What is left needs one: whether these numbers match the Python's."
echo "    That is scripts/test.sh's parity example, with a recorded response."
