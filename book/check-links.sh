#!/usr/bin/env bash
# Check the links in the built book with lychee (https://lychee.cli.rs).
#
# Usage: book/check-links.sh [--offline] [BUILT_BOOK_DIR]
#   BUILT_BOOK_DIR defaults to book/book (run `mdbook build book` first).
#   --offline: check only the links between pages of the book.
#   Default:   also check the links to docs.rs (these follow the last release, not this branch).
#
# The site is served under /diffsol/, and some pages use root-relative links
# (e.g. /diffsol/primer/...). We link the built book to <tmp>/diffsol so that lychee can
# resolve these links with --root-dir.
#
# Not checked: external sites other than docs.rs (they are often slow or block CI), and
# print.html (mdBook's print page breaks root-relative links, see the primer index page).
set -euo pipefail

offline=()
if [ "${1:-}" = "--offline" ]; then
    offline=(--offline)
    shift
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
book_dir="$(cd "${1:-$script_dir/book}" && pwd)"

if [ ! -f "$book_dir/index.html" ]; then
    echo "error: no built book in $book_dir (run 'mdbook build book' first)" >&2
    exit 1
fi

site_dir="$(mktemp -d)"
trap 'rm -rf "$site_dir"' EXIT
ln -s "$book_dir" "$site_dir/diffsol"

lychee \
    ${offline[@]+"${offline[@]}"} \
    --no-progress \
    --root-dir "$site_dir" \
    --exclude-path 'print\.html$' \
    --include '^file://' \
    --include '^https://docs\.rs/' \
    --max-concurrency 8 \
    "$site_dir/diffsol"
