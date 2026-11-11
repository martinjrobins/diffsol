#!/usr/bin/env bash
# Compile and run the C examples against a Cargo build of diffsol-c.
#
# Usage: run_examples.sh [<dir>]
#
# <dir> is the directory where `cargo build -p diffsol-c` put the library and include/diffsol.h
# (default: target/debug). The script compiles the DiffSL JIT example only if the library has a
# JIT backend, and also runs it with LLVM if the library has LLVM. Each example returns a
# non-zero exit code if its solution is not correct.
#
# On Windows (Git Bash), the script uses MSVC (`cl`), so run it in a developer environment. On
# other systems it uses $CC, or `cc`.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
lib=$(cd "${1:-target/debug}" && pwd)
header="$lib/include/diffsol.h"
if [ ! -f "$header" ]; then
    echo "error: $header does not exist; build diffsol-c with cargo first" >&2
    exit 1
fi

# whether the header (and so the library) has the feature macro $1
has() { grep -q "^#define $1 1" "$header"; }

examples=(logistic_callbacks)
if has DIFFSOL_DIFFSL_CRANELIFT || has DIFFSOL_DIFFSL_LLVM; then
    examples+=(logistic_diffsl_jit)
fi

out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT

windows=false
if [ "${OS:-}" = "Windows_NT" ]; then
    windows=true
    # Windows finds the DLL with PATH
    export PATH="$lib:$PATH"
    # Git Bash has its own `link`, which can come before the MSVC linker in PATH; put the
    # MSVC tools first (the developer environment sets these variables)
    if [ -n "${VCToolsInstallDir:-}" ]; then
        msvc_bin="$(cygpath -u "$VCToolsInstallDir")bin/Host${VSCMD_ARG_HOST_ARCH:-x64}/${VSCMD_ARG_TGT_ARCH:-x64}"
        export PATH="$msvc_bin:$PATH"
    fi
fi

for example in "${examples[@]}"; do
    echo "== $example"
    if $windows; then
        # MSVC needs Windows paths; options with `-` are not converted by Git Bash
        (cd "$out" && cl -nologo -W3 "$(cygpath -w "$here/$example.c")" \
            "-I$(cygpath -w "$lib/include")" "$(cygpath -w "$lib/diffsol_c.dll.lib")" \
            "-Fe$example.exe")
        exe="$out/$example.exe"
    else
        "${CC:-cc}" -std=c11 -Wall -Wextra -Werror "$here/$example.c" -I"$lib/include" \
            -L"$lib" -ldiffsol_c -lm -Wl,-rpath,"$lib" -o "$out/$example"
        exe="$out/$example"
    fi
    "$exe"
done

# the JIT example uses Cranelift if the library has it; also run it with LLVM
if has DIFFSOL_DIFFSL_LLVM; then
    echo "== logistic_diffsl_jit (LLVM)"
    DIFFSOL_JIT_BACKEND=llvm "$out/logistic_diffsl_jit$($windows && echo .exe || true)"
fi
