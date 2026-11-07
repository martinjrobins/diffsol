#!/bin/sh
# Write the files of the Debian packaging that depend on the crate version:
# debian/control, debian/changelog, debian/soversion and the .install file of the runtime
# package libdiffsol<soversion>. Run this before `dpkg-buildpackage`.
#
# Environment variables:
# - DEB_CODENAME: the codename of the distribution to build for, for example bookworm or noble.
#   It is added to the package version (<version>-<revision>~<codename>), so that the packages of
#   different distributions have different versions. Default: no codename, distribution unstable.
# - LLVM_VERSION: the major version of the LLVM package of the distribution to link with
#   (default 19). The library links it as a shared library.
# - DEB_REVISION: the Debian revision (default 1).
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

llvm="${LLVM_VERSION:-19}"
revision="${DEB_REVISION:-1}"
if [ -n "${DEB_CODENAME:-}" ]; then
    deb_version="$version-$revision~$DEB_CODENAME"
    distribution="$DEB_CODENAME"
else
    deb_version="$version-$revision"
    distribution="unstable"
fi

echo "$soversion" > soversion
echo "$llvm" > llvm-version
sed -e "s/@SOVERSION@/$soversion/g" -e "s/@LLVM@/$llvm/g" control.in > control
echo "usr/lib/*/libdiffsol.so.*" > "libdiffsol$soversion.install"

cat > changelog <<EOF
diffsol ($deb_version) $distribution; urgency=medium

  * Release $version of diffsol-c.

 -- Martin Robinson <martinjrobins@gmail.com>  $(LC_ALL=C date -u '+%a, %d %b %Y %H:%M:%S +0000')
EOF

echo "diffsol $deb_version: runtime package libdiffsol$soversion, LLVM $llvm"
