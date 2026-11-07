#!/bin/sh
# Write the files of the Debian packaging that depend on the crate version:
# debian/control, debian/changelog, debian/soversion and the .install file of the runtime
# package libdiffsol<soversion>. Run this before `dpkg-buildpackage`. The environment variable
# DEB_REVISION sets the Debian revision (default 1).
set -eu

cd "$(dirname "$0")"

version=$(sed -n 's/^version = "\(.*\)"$/\1/p' ../Cargo.toml | head -n 1)
if [ -z "$version" ]; then
    echo "cannot find the version in Cargo.toml" >&2
    exit 1
fi
major=$(echo "$version" | cut -d . -f 1)
minor=$(echo "$version" | cut -d . -f 2)

# The soversion is the version in the SONAME (libdiffsol.so.<soversion>) and in the name of the
# runtime package. Before version 1.0, a minor release can break the ABI.
if [ "$major" = "0" ]; then
    soversion="$major.$minor"
else
    soversion="$major"
fi

echo "$soversion" > soversion
sed "s/@SOVERSION@/$soversion/g" control.in > control
echo "usr/lib/*/libdiffsol.so.*" > "libdiffsol$soversion.install"

cat > changelog <<EOF
diffsol ($version-${DEB_REVISION:-1}) unstable; urgency=medium

  * Release $version of diffsol-c.

 -- Martin Robinson <martinjrobins@gmail.com>  $(LC_ALL=C date -u '+%a, %d %b %Y %H:%M:%S +0000')
EOF

echo "diffsol $version: runtime package libdiffsol$soversion"
