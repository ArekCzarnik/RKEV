#!/usr/bin/env bash
#
# Run the triage example against a Kev server.
#
#   scripts/triage.sh                              # the README ticket
#   scripts/triage.sh "My parcel never arrived."   # your own ticket
#   scripts/triage.sh -f ticket.txt                # ticket from a file
#   cat ticket.txt | scripts/triage.sh -           # ticket from stdin
#   scripts/triage.sh --release                    # optimised build
#   scripts/triage.sh --no-check                   # skip the server probe
#
# Reads KEV_BASE_URL (default http://127.0.0.1:8009), KEV_MODEL and
# KEV_API_KEY, which the example itself picks up from the environment.

set -euo pipefail

CRATE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/kev-client"
BASE_URL="${KEV_BASE_URL:-http://127.0.0.1:8009}"
SERVE_HINT="uv run --extra serve python -m kev.serve --run jaredpalmer/kev-4b --port 8009"

ticket=""
ticket_file=""
check=1
cargo_flags=()

while [ $# -gt 0 ]; do
    case "$1" in
        -f|--file)  ticket_file="${2:-}"; shift 2 ;;
        -)          ticket_file="/dev/stdin"; shift ;;
        --no-check) check=0; shift ;;
        --release)  cargo_flags+=(--release); shift ;;
        -h|--help)  sed -n '3,13p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)          ticket="$1"; shift ;;
    esac
done

if [ -n "$ticket_file" ]; then
    [ -r "$ticket_file" ] || { echo "error: cannot read $ticket_file" >&2; exit 1; }
    ticket="$(cat "$ticket_file")"
elif [ -z "$ticket" ] && { [ -p /dev/stdin ] || [ -f /dev/stdin ]; }; then
    # Only a real pipe or file redirect. "not a tty" would also be true under
    # cron, CI and nohup, where reading stdin blocks forever.
    ticket="$(cat)"
fi

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

# Probe the server before building: a compile takes a while and the failure
# is far clearer here than as a connection error after it.
if [ "$check" -eq 1 ] && command -v curl >/dev/null 2>&1; then
    # `${a[@]+"${a[@]}"}` rather than a bare `"${a[@]}"`: bash before 4.4 -
    # which is what /bin/bash is on macOS - treats an empty array as unset under
    # `set -u` and aborts the script.
    auth=()
    [ -n "${KEV_API_KEY:-}" ] && auth=(-H "Authorization: Bearer ${KEV_API_KEY}")
    # curl already prints 000 on a connection failure; the ||-branch covers
    # curl being killed before it prints anything at all.
    status="$(curl -sS -o /dev/null -w '%{http_code}' --max-time 5 \
        ${auth[@]+"${auth[@]}"} "${BASE_URL%/}/v1/models" 2>/dev/null)" || status="000"
    case "$status" in
        2??) echo "==> kev server up at $BASE_URL" ;;
        401|403)
            echo "error: $BASE_URL needs a bearer token - set KEV_API_KEY." >&2
            exit 1 ;;
        000)
            echo "error: no kev server answering at $BASE_URL" >&2
            echo "       start one with:  $SERVE_HINT" >&2
            echo "       or point KEV_BASE_URL elsewhere, or pass --no-check" >&2
            exit 1 ;;
        *)  echo "warning: $BASE_URL/v1/models returned HTTP $status - trying anyway" >&2 ;;
    esac
fi

cd "$CRATE_DIR"

echo "==> cargo run --example triage${KEV_MODEL:+  (model ${KEV_MODEL})}"
if [ -n "$ticket" ]; then
    exec cargo run ${cargo_flags[@]+"${cargo_flags[@]}"} --example triage -- "$ticket"
else
    exec cargo run ${cargo_flags[@]+"${cargo_flags[@]}"} --example triage
fi
