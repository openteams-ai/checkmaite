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

artifact_directory="artifacts/container/${artifact_variant}"
build_metadata="${artifact_directory}/build-metadata.json"
push_metadata="${artifact_directory}/push-metadata.json"
test -s "${build_metadata}"

# The scanned image is the one the build step loaded into the daemon. Its config
# digest covers its configuration and every layer's uncompressed digest.
scanned_config=$(jq -r '."containerimage.config.digest" // empty' "${build_metadata}")
printf '%s\n' "${scanned_config}" | grep -Eq '^sha256:[0-9a-f]{64}$'
revision=$(docker image inspect \
    --format '{{index .Config.Labels "org.opencontainers.image.revision"}}' \
    "${local_image}")
test -n "${revision}"

# Push a pipeline-scoped staging tag so the digest exists in the registry for
# Cosign. The Semantic Version tag is applied here only after verification.
# `docker push` from the daemon's image store re-encodes the image with Docker
# media types, which CS-1-H-1 rejects, so BuildKit pushes the same, fully cached
# build with OCI media types instead.
sh "${script_directory}/build_container.sh" \
    "${repository_variant}" "${release_tag}" "${revision}" "${push_metadata}" \
    --output "type=image,name=${staging_reference},push=true,oci-mediatypes=true"

pushed_digest=$(jq -r '."containerimage.digest" // empty' "${push_metadata}")
release_digest="${harbor_image_base}@${pushed_digest}"
harbor_encode_digest "${release_digest}" >/dev/null

cleanup_staging() {
    harbor_remove_tag "${release_digest}" "${staging_tag}" || true
}
# Exit on a signal so the EXIT trap removes staging tags without resuming the
# signing flow.
trap cleanup_staging EXIT
trap 'exit 1' HUP INT TERM

# Read the manifest back from Harbor: it must be an OCI image manifest for the
# exact image that was tested and scanned.
published_manifest="${artifact_directory}/published-manifest.json"
docker buildx imagetools inspect --raw "${release_digest}" > "${published_manifest}"
published_media_type=$(jq -r '.mediaType' "${published_manifest}")
published_config=$(jq -r '.config.digest' "${published_manifest}")
if [ "${published_media_type}" != "application/vnd.oci.image.manifest.v1+json" ]; then
    echo "${release_digest} has media type ${published_media_type}, not an OCI image manifest" >&2
    exit 1
fi
if [ "${published_config}" != "${scanned_config}" ]; then
    echo "pushed config ${published_config} does not match scanned image config ${scanned_config}" >&2
    exit 1
fi

cosign_key="awskms:///${COSIGN_KMS_KEY}"
signing_config=docker/ci/cosign-signing-config.json
sbom="${artifact_directory}/sbom.spdx.json"
vulnerability_report="${artifact_directory}/vulnerability-attestation.json"
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
