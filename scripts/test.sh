#!/usr/bin/env bash
#
# Run the kev-client checks: formatting, lints, then tests.
#
#   scripts/test.sh                                  # everything
#   scripts/test.sh request_matches_the_readme_example   # one test by name
#   scripts/test.sh --test wire_format               # one test file
#   scripts/test.sh --skip-fmt --skip-clippy         # tests only
#   scripts/test.sh --skip-features                  # skip the feature matrix
#
# Extra arguments are passed through to `cargo test`.
# The tests need no running Kev server.

set -euo pipefail

CRATE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/kev-client"

skip_fmt=0
skip_clippy=0
skip_features=0
cargo_test_args=()

for arg in "$@"; do
    case "$arg" in
        --skip-fmt)    skip_fmt=1 ;;
        --skip-clippy) skip_clippy=1 ;;
        --skip-features) skip_features=1 ;;
        -h|--help)     sed -n '3,13p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)             cargo_test_args+=("$arg") ;;
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

if [ ! -f "$CRATE_DIR/Cargo.toml" ]; then
    echo "error: no crate at $CRATE_DIR - run this script from inside the repo." >&2
    exit 1
fi

cd "$CRATE_DIR"

echo "==> $(cargo --version)  in  $CRATE_DIR"

# Every step runs even when an earlier one failed, and the failures are
# summarised at the end: a formatting nit must not hide whether the crate
# still compiles. The exit code is still non-zero if anything failed.
failed=""

step() {
    local label="$1"
    shift
    echo "==> $label"
    if ! "$@"; then
        failed="${failed}${failed:+, }${label}"
    fi
}

# clippy runs with --all-features: a lint inside a cfg-gated module would
# otherwise never be seen, which is how the manual_async_fn in local.rs got
# past the first run.
#
# fmt and clippy ship as optional rustup components; skip them with a note
# rather than failing a run that could still execute the tests.
if [ "$skip_fmt" -eq 0 ]; then
    if cargo fmt --version >/dev/null 2>&1; then
        step "cargo fmt --check" cargo fmt --all -- --check
    else
        echo "==> skipping fmt (rustfmt not installed: rustup component add rustfmt)"
    fi
fi

if [ "$skip_clippy" -eq 0 ]; then
    if cargo clippy --version >/dev/null 2>&1; then
        step "cargo clippy" cargo clippy --all-targets --all-features -- -D warnings
    else
        echo "==> skipping clippy (not installed: rustup component add clippy)"
    fi
fi

# `${a[@]+"${a[@]}"}` rather than a bare `"${a[@]}"`: bash before 4.4 - which
# is what /bin/bash is on macOS - treats an empty array as unset under `set -u`
# and aborts the script.
step "cargo test" cargo test ${cargo_test_args[@]+"${cargo_test_args[@]}"}

# http and local are optional, so every combination has to build on its own -
# a cfg that only compiles with default features is a trap that shows up much
# later. Skipped when a specific test was named, where the matrix is noise.
if [ "$skip_features" -eq 0 ] && [ ${#cargo_test_args[@]} -eq 0 ]; then
    for combo in "--no-default-features" \
                 "--no-default-features --features local" \
                 "--all-features"; do
        # shellcheck disable=SC2086
        step "cargo test $combo" cargo test $combo
    done
fi

if [ -n "$failed" ]; then
    echo ""
    echo "==> FAILED: $failed" >&2
    exit 1
fi

echo "==> all checks passed"
