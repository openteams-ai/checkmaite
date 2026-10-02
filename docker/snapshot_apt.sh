#!/bin/bash
# Bring an Ubuntu 24.04 image to one archive snapshot and install packages.
#
#   snapshot_apt.sh SNAPSHOT [PACKAGE...]
#
# Every source is replaced by snapshot.ubuntu.com at SNAPSHOT (UTC,
# YYYYMMDDTHHMMSSZ), so package versions are fixed until SNAPSHOT moves. Any
# index that fails to download fails the build. NVIDIA's apt source is removed,
# so CUDA, cuDNN, and NCCL packages never change.
set -euo pipefail

if [[ "$#" -lt 1 || ! "$1" =~ ^[0-9]{8}T[0-9]{6}Z$ ]]; then
    echo "usage: $0 YYYYMMDDTHHMMSSZ [PACKAGE...]" >&2
    exit 2
fi
snapshot=$1
shift

case "$(dpkg --print-architecture)" in
    amd64) archive=ubuntu ;;
    *) archive=ubuntu-ports ;;
esac

rm -f /etc/apt/sources.list /etc/apt/sources.list.d/*
cat > /etc/apt/sources.list.d/snapshot.sources <<SOURCES
Types: deb
URIs: https://snapshot.ubuntu.com/${archive}/${snapshot}/
Suites: noble noble-updates noble-security
Components: main universe
Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg
SOURCES

export DEBIAN_FRONTEND=noninteractive
apt_options=(-o APT::Update::Error-Mode=any -o APT::Install-Recommends=false)

# The plain Ubuntu image has no CA bundle, and snapshot.ubuntu.com is HTTPS
# only. Fetch ca-certificates once without TLS peer verification; apt still
# verifies every index and package against the archive signing key from the
# digest-pinned base, which is the trust model of Ubuntu's default HTTP
# mirrors. All later operations verify TLS.
if [[ ! -s /etc/ssl/certs/ca-certificates.crt ]]; then
    apt-get "${apt_options[@]}" -o Acquire::https::Verify-Peer=false update
    apt-get "${apt_options[@]}" -o Acquire::https::Verify-Peer=false \
        install -y ca-certificates
fi

apt-get "${apt_options[@]}" update
apt-get "${apt_options[@]}" upgrade -y
if [[ "$#" -gt 0 ]]; then
    apt-get "${apt_options[@]}" install -y "$@"
fi
apt-get clean
rm -rf /var/lib/apt/lists/*
