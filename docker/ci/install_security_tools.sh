#!/bin/sh
set -eu

SYFT_VERSION=1.52.0
SYFT_ARCHIVE="syft_${SYFT_VERSION}_linux_amd64.tar.gz"
SYFT_SHA256=caeedb81fb0491615f1ebd1761e4145d41ee86dd2cc7bf80669f9f5ad9d6133d
TRIVY_VERSION=0.74.0
TRIVY_ARCHIVE="trivy_${TRIVY_VERSION}_Linux-64bit.tar.gz"
TRIVY_SHA256=2ae6fe3ee734b7fdf11335663e18c75ea12dccc76062f09f164a3b0f8be4371a
TRIVY_TEMPLATE_SHA256=6921a9ba0ac4f5ed0c8f4bb6a0c23465b907e76dd5f7c7920f70fa3446523a63
TOOLS_DIR=${TOOLS_DIR:-/usr/local/bin}
TRIVY_SHARE_DIR=${TRIVY_SHARE_DIR:-/usr/local/share/trivy}
mkdir -p "${TOOLS_DIR}" "${TRIVY_SHARE_DIR}"

install_archive() {
    name=$1
    version=$2
    archive=$3
    digest=$4
    repository=$5

    curl --fail --location --silent --show-error \
        "https://github.com/${repository}/releases/download/v${version}/${archive}" \
        --output "/tmp/${archive}"
    actual_digest=$(sha256sum "/tmp/${archive}" | awk '{print $1}')
    test "${actual_digest}" = "${digest}"
    tar -xzf "/tmp/${archive}" -C "${TOOLS_DIR}" "${name}"
    rm -f "/tmp/${archive}"
}

install_archive syft "${SYFT_VERSION}" "${SYFT_ARCHIVE}" "${SYFT_SHA256}" anchore/syft
install_archive trivy "${TRIVY_VERSION}" "${TRIVY_ARCHIVE}" "${TRIVY_SHA256}" aquasecurity/trivy

curl --fail --location --silent --show-error \
    "https://raw.githubusercontent.com/aquasecurity/trivy/v${TRIVY_VERSION}/contrib/gitlab.tpl" \
    --output "${TRIVY_SHARE_DIR}/gitlab.tpl"
template_digest=$(sha256sum "${TRIVY_SHARE_DIR}/gitlab.tpl" | awk '{print $1}')
test "${template_digest}" = "${TRIVY_TEMPLATE_SHA256}"

"${TOOLS_DIR}/syft" version
"${TOOLS_DIR}/trivy" version
