#!/bin/sh
set -eu

script_directory=$(dirname -- "$0")
script_directory=$(cd -- "${script_directory}" && pwd)
# shellcheck disable=SC1091
. "${script_directory}/release.sh"
# shellcheck disable=SC1091
. "${script_directory}/harbor.sh"
harbor_image_base=

if [ "$#" -ne 5 ]; then
    echo "usage: $0 LOCAL_IMAGE cpu|cuda ARTIFACT_VARIANT SEMVER release" >&2
    exit 2
fi

local_image=$1
repository_variant=$2
artifact_variant=$3
release_tag=$4
publish_mode=$5

case "${repository_variant}:${artifact_variant}:${publish_mode}" in
    cpu:cpu-amd64:release|cuda:cuda:release) ;;
    *)
        echo "unsupported repository, artifact variant, and publication mode" >&2
        exit 2
        ;;
esac

validate_release_tag "${release_tag}"
harbor_configure_repository "${repository_variant}"
: "${COSIGN_KMS_KEY:?COSIGN_KMS_KEY is required}"
: "${CI_PIPELINE_ID:?CI_PIPELINE_ID is required}"

release_reference="${harbor_image_base}:${release_tag}"
harbor_assert_release_tag_available "${release_tag}"
staging_tag=$(harbor_staging_tag "${release_tag}" "${artifact_variant}")
staging_reference="${harbor_image_base}:${staging_tag}"

# Push a pipeline-scoped staging tag so the digest exists in the registry for
# Cosign. The Semantic Version tag is applied here only after verification.
docker tag "${local_image}" "${staging_reference}"
docker push "${staging_reference}"

release_digest=$(
    docker image inspect --format '{{range .RepoDigests}}{{println .}}{{end}}' \
        "${staging_reference}" \
        | grep "^${harbor_image_base}@sha256:" \
        | head -n 1
)
test -n "${release_digest}"
harbor_encode_digest "${release_digest}" >/dev/null

cleanup_staging() {
    harbor_remove_tag "${release_digest}" "${staging_tag}" || true
}
# Exit on a signal so the EXIT trap removes staging tags without resuming the
# signing flow.
trap cleanup_staging EXIT
trap 'exit 1' HUP INT TERM

cosign_key="awskms:///${COSIGN_KMS_KEY}"
signing_config=docker/ci/cosign-signing-config.json
sbom="artifacts/container/${artifact_variant}/sbom.spdx.json"
vulnerability_report="artifacts/container/${artifact_variant}/vulnerability-attestation.json"
test -s "${signing_config}"
test "$(jq -cS . "${signing_config}")" = \
    '{"mediaType":"application/vnd.dev.sigstore.signingconfig.v0.2+json","rekorTlogConfig":{},"tsaConfig":{}}'
test -s "${sbom}"
test -s "${vulnerability_report}"

cosign sign --yes \
    --signing-config "${signing_config}" \
    --key "${cosign_key}" \
    "${release_digest}"
cosign attest --yes \
    --signing-config "${signing_config}" \
    --key "${cosign_key}" \
    --type spdxjson \
    --predicate "${sbom}" \
    "${release_digest}"
cosign attest --yes \
    --signing-config "${signing_config}" \
    --key "${cosign_key}" \
    --type vuln \
    --predicate "${vulnerability_report}" \
    "${release_digest}"
cosign verify --insecure-ignore-tlog=true \
    --key "${cosign_key}" "${release_digest}" >/dev/null
cosign verify-attestation --insecure-ignore-tlog=true \
    --key "${cosign_key}" \
    --type spdxjson \
    "${release_digest}" >/dev/null
cosign verify-attestation --insecure-ignore-tlog=true \
    --key "${cosign_key}" \
    --type vuln \
    "${release_digest}" >/dev/null

artifact_directory="artifacts/container/${artifact_variant}"
mkdir -p "${artifact_directory}"
printf '%s\n' "${release_digest}" \
    > "${artifact_directory}/published-digest.txt"
if [ "${CONTAINER_RELEASE_DRY_RUN:-false}" = "true" ]; then
    echo "Verified dry-run container ${release_digest}; no release tag was applied"
    exit 0
fi

# This is the final fallible publication action. A failed signing or attestation
# can never strand an unsigned, immutable Semantic Version tag.
harbor_add_tag "${release_digest}" "${release_tag}"

echo "Published signed immutable container ${release_digest} as ${release_reference}"
