#!/bin/bash
# Build and check an explicitly owned macOS benchmark copy; original untouched.
set -euo pipefail
build_dir="${1:?Usage: build_no_willneed.sh ABSOLUTE_BUILD_DIRECTORY [ORIGINAL_LLAMA_SERVER]}"
original="${2:-/Applications/Ollama.app/Contents/Resources/llama-server}"
source_dir="$(cd "$(dirname "$0")" && pwd)"
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 || "$build_dir" != /* ]]; then
    echo 'Requires Apple Silicon macOS and an absolute owned build directory.' >&2
    exit 2
fi
mkdir -p "$build_dir"
cc -arch arm64 -dynamiclib -O2 -Wall -Wextra -Werror \
    "$source_dir/no_willneed.c" -o "$build_dir/libairllm_no_willneed.dylib"
cc -arch arm64 -O2 -Wall -Wextra -Werror \
    "$source_dir/test_no_willneed.c" -o "$build_dir/test_no_willneed"
"$build_dir/test_no_willneed" > "$build_dir/baseline.json"
DYLD_INSERT_LIBRARIES="$build_dir/libairllm_no_willneed.dylib" \
    "$build_dir/test_no_willneed" > "$build_dir/interposed.json" 2> "$build_dir/interposed.log"
cmp "$build_dir/baseline.json" "$build_dir/interposed.json"
if [[ "$(lipo -archs "$original")" == arm64 ]]; then
    cp "$original" "$build_dir/llama-server-owned"
else
    lipo "$original" -thin arm64 -output "$build_dir/llama-server-owned"
fi
# The original application is hardened. Only this private extracted copy gets
# an ad-hoc signature permitting this process's explicitly supplied DYLD shim.
codesign --force --sign - --options 0 "$build_dir/llama-server-owned"
DYLD_INSERT_LIBRARIES="$build_dir/libairllm_no_willneed.dylib" \
    "$build_dir/llama-server-owned" --version > "$build_dir/owned-version.log" 2>&1
echo "Owned binary: $build_dir/llama-server-owned"
echo "Process-only DYLD library: $build_dir/libairllm_no_willneed.dylib"
