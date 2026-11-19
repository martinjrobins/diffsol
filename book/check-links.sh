#!/usr/bin/env bash
# Check the links in the built book with lychee (https://lychee.cli.rs).
#
# Usage: book/check-links.sh [--offline] [BUILT_BOOK_DIR]
#   BUILT_BOOK_DIR defaults to book/book (run `mdbook build book` first).
#   --offline: check only the links between pages of the book.
#   Default:   also check the links to docs.rs (these follow the last release, not this branch).
#
# Not checked: external sites other than docs.rs (they are often slow or block CI), and
# 404.html (mdBook gives it a root-relative <base href>, which only works on the served site).
set -euo pipefail

offline=()
if [ "${1:-}" = "--offline" ]; then
    offline=(--offline)
    shift
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
book_dir="${1:-$script_dir/book}"

if [ ! -f "$book_dir/index.html" ]; then
    echo "error: no built book in $book_dir (run 'mdbook build book' first)" >&2
    exit 1
fi

lychee \
    ${offline[@]+"${offline[@]}"} \
    --no-progress \
    --exclude-path '404\.html$' \
    --include '^file://' \
    --include '^https://docs\.rs/' \
    --max-concurrency 8 \
    "$book_dir"
