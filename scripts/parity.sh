#!/usr/bin/env bash
#
# Record what the Python Kev server answers, then hold the engine to it.
#
# This is the one check that needs the server, and it needs it once: the
# recordings it writes are files, so --check-only repeats the comparison offline
# for as long as they are kept.
#
#   scripts/parity.sh --base <dir> --checkpoint <dir>        # record, then compare
#   scripts/parity.sh --base <dir> --checkpoint <dir> --check-only
#                                                           # compare recordings again
#   scripts/parity.sh --record-only                         # only talk to the server
#
#   --url <url>          the server (default http://127.0.0.1:8009)
#   --requests <dir>     requests to send (default rkev/tests/parity)
#   --recordings <dir>   where responses go (default <requests>/recordings)
#   --tolerance <f64>    largest probability difference to accept (default 0.01)
#   --only <name>        one request by file stem, repeatable
#
# The server has to be serving the same checkpoint the engine loads, and
# KEV_DTYPE=fp32 is the path the published numbers were measured on:
#
#   KEV_DTYPE=fp32 uv run --extra serve python -m kev.serve \
#       --run jaredpalmer/kev-0.6b --port 8009
#
# KEV_API_KEY is sent as a bearer token if it is set.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CRATE_DIR="$ROOT/rkev"
URL="${KEV_BASE_URL:-http://127.0.0.1:8009}"
REQUESTS="$CRATE_DIR/tests/parity"
RECORDINGS=""
TOLERANCE="0.01"

base=""
checkpoint=""
head=""
record=1
check=1
only=()

while [ $# -gt 0 ]; do
    case "$1" in
        --base)        base="${2:-}"; shift 2 ;;
        --checkpoint)  checkpoint="${2:-}"; shift 2 ;;
        --head)        head="${2:-}"; shift 2 ;;
        --url)         URL="${2:-}"; shift 2 ;;
        --requests)    REQUESTS="${2:-}"; shift 2 ;;
        --recordings)  RECORDINGS="${2:-}"; shift 2 ;;
        --tolerance)   TOLERANCE="${2:-}"; shift 2 ;;
        --only)        only+=("${2:-}"); shift 2 ;;
        --record-only) check=0; shift ;;
        --check-only)  record=0; shift ;;
        -h|--help)     sed -n '3,27p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)             echo "error: unknown argument $1 (try --help)" >&2; exit 1 ;;
    esac
done

[ -n "$RECORDINGS" ] || RECORDINGS="$REQUESTS/recordings"
[ -d "$REQUESTS" ] || { echo "error: no request directory $REQUESTS" >&2; exit 1; }

# rustup installs into ~/.cargo but only a login shell picks that up.
if [ "$check" -eq 1 ]; then
    if ! command -v cargo >/dev/null 2>&1 && [ -f "${CARGO_HOME:-$HOME/.cargo}/env" ]; then
        # shellcheck disable=SC1091
        . "${CARGO_HOME:-$HOME/.cargo}/env"
    fi
    if ! command -v cargo >/dev/null 2>&1; then
        echo "error: cargo is not on PATH - no Rust toolchain in this environment." >&2
        exit 127
    fi
    [ -n "$base" ] || { echo "error: --base is required to compare (not for --record-only)" >&2; exit 1; }
fi

# Which requests, and under which name each recording lands.
names=()
for path in "$REQUESTS"/*.json; do
    [ -e "$path" ] || continue
    name="$(basename "$path" .json)"
    if [ ${#only[@]} -gt 0 ]; then
        wanted=0
        for pick in ${only[@]+"${only[@]}"}; do
            [ "$pick" = "$name" ] && wanted=1
        done
        [ "$wanted" -eq 1 ] || continue
    fi
    names+=("$name")
done
[ ${#names[@]} -gt 0 ] || { echo "error: no requests to run in $REQUESTS" >&2; exit 1; }

failed=""
note() { failed="${failed}${failed:+, }$1"; }

# ---------------------------------------------------------------------------
# Record: ask the server, keep the answer
# ---------------------------------------------------------------------------

if [ "$record" -eq 1 ]; then
    command -v curl >/dev/null 2>&1 || { echo "error: recording needs curl" >&2; exit 1; }
    auth=()
    [ -n "${KEV_API_KEY:-}" ] && auth=(-H "Authorization: Bearer ${KEV_API_KEY}")

    # Probe before anything else: a compile takes a while and "connection
    # refused" is far clearer here than after it.
    status="$(curl -sS -o /dev/null -w '%{http_code}' --max-time 5 \
        ${auth[@]+"${auth[@]}"} "${URL%/}/v1/models" 2>/dev/null)" || status="000"
    case "$status" in
        2??) echo "==> kev server up at $URL" ;;
        401|403) echo "error: $URL wants a bearer token - set KEV_API_KEY." >&2; exit 1 ;;
        000)
            echo "error: no kev server answering at $URL" >&2
            echo "       start one with:" >&2
            echo "       KEV_DTYPE=fp32 uv run --extra serve python -m kev.serve \\" >&2
            echo "           --run jaredpalmer/kev-0.6b --port 8009" >&2
            exit 1 ;;
        *) echo "warning: $URL/v1/models returned HTTP $status - trying anyway" >&2 ;;
    esac

    mkdir -p "$RECORDINGS"
    # What the server is serving, kept beside the recordings: a pair recorded
    # against a different checkpoint than the engine loads is worse than none.
    curl -sS ${auth[@]+"${auth[@]}"} "${URL%/}/v1/models" > "$RECORDINGS/models.json" || true

    for name in "${names[@]}"; do
        for endpoint in systemone systemone/separate; do
            suffix=""
            [ "$endpoint" = "systemone/separate" ] && suffix=".separate"
            out="$RECORDINGS/$name$suffix.response.json"
            code="$(curl -sS -o "$out" -w '%{http_code}' --max-time 300 \
                -H 'content-type: application/json' ${auth[@]+"${auth[@]}"} \
                -d @"$REQUESTS/$name.json" "${URL%/}/v1/$endpoint" 2>/dev/null)" || code="000"
            case "$code" in
                2??) echo "    recorded $name$suffix" ;;
                *)
                    echo "    FAILED   $name$suffix (HTTP $code): $(head -c 200 "$out")" >&2
                    rm -f "$out"
                    note "recording $name$suffix" ;;
            esac
        done
    done
fi

# ---------------------------------------------------------------------------
# Compare: the same requests, in process
# ---------------------------------------------------------------------------

if [ "$check" -eq 1 ]; then
    cd "$CRATE_DIR"
    echo ""
    echo "==> comparing, tolerance $TOLERANCE"
    if [ -f "$RECORDINGS/models.json" ]; then
        echo "    the server was serving: $(head -c 200 "$RECORDINGS/models.json")"
    fi

    model=(--base "$base")
    [ -n "$checkpoint" ] && model+=(--checkpoint "$checkpoint")
    [ -n "$head" ] && model+=(--head "$head")

    for name in "${names[@]}"; do
        for suffix in "" ".separate"; do
            recording="$RECORDINGS/$name$suffix.response.json"
            [ -f "$recording" ] || continue
            flag=()
            [ "$suffix" = ".separate" ] && flag=(--separate)
            echo ""
            echo "--- $name$suffix"
            if ! cargo run -q --release --example parity -- \
                "${model[@]}" ${flag[@]+"${flag[@]}"} \
                --request "$REQUESTS/$name.json" --server "$recording" \
                --tolerance "$TOLERANCE"; then
                note "$name$suffix"
            fi
        done
    done
fi

echo ""
if [ -n "$failed" ]; then
    echo "==> FAILED: $failed" >&2
    echo "    A difference here is the engine's, not the server's. Read the" >&2
    echo "    token counts first: if input_tokens differ, the prompt differs," >&2
    echo "    and the probabilities are downstream of that." >&2
    exit 1
fi
echo "==> the engine matches the server on every recorded request."
echo "    The recordings are files: scripts/parity.sh --check-only repeats this"
echo "    from the recordings alone, so keep them."
