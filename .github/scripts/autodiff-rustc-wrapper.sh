#!/bin/sh

rustc="$1"
shift

if [ "$2" = "diffsol" ]; then
    exec "$rustc" "$@" -Zautodiff=Enable
fi

exec "$rustc" "$@"
