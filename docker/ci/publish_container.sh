#!/bin/sh
# Sign, attest, and verify the digest that stage_container.sh pushed, then apply
# the write-once Semantic Version tag through the registry API. Runs on k8s-small-cpu, the only runner
# class with the AWS KMS signing permissions; it needs no Docker daemon.
set -eu

script_directory=$(dirname -- "$0")
script_directory=$(cd -- "${script_directory}" && pwd)
# shellcheck disable=SC1091
. "${script_directory}/release.sh"
# shellcheck disable=SC1091
. "${script_directory}/harbor.sh"
harbor_image_base=

if [ "$#" -ne 4 ]; then
    echo "usage: $0 cpu|cuda ARTIFACT_VARIANT SEMVER release" >&2
    exit 2
fi

repository_variant=$1
artifact_variant=$2
release_tag=$3
publish_mode=$4

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

artifact_directory="artifacts/container/${artifact_variant}"
release_reference="${harbor_image_base}:${release_tag}"
# Another pipeline may have published this version since the staging job ran.
harbor_assert_release_tag_available "${release_tag}"

release_digest=$(cat "${artifact_directory}/staging-digest.txt")
case "${release_digest}" in
    "${harbor_image_base}@sha256:"*) ;;
    *)
        echo "staged digest is not in ${harbor_image_base}: ${release_digest}" >&2
        exit 1
        ;;
esac
harbor_digest "${release_digest}" >/dev/null

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
    # Apply a throwaway staging tag the same way as the release tag, so the dry
    # run exercises the push token, the manifest PUT, and the check that the tag
    # resolves to the signed digest. It is not a Semantic Version, so the
    # immutability rule does not apply to it.
    dry_run_tag="staging-dryrun-${CI_PIPELINE_ID}-${artifact_variant}"
    harbor_validate_tag "${dry_run_tag}"
    harbor_add_tag "${release_digest}" "${dry_run_tag}" \
        "${artifact_directory}/release-manifest.json"
    echo "Verified dry-run container ${release_digest} and tagged it ${harbor_image_base}:${dry_run_tag}; no release tag was applied"
    exit 0
fi

# This is the final fallible publication action. A failed signing or attestation
# can never strand an unsigned, immutable Semantic Version tag. The tag is
# applied by pushing the signed digest's exact manifest bytes under it, and is
# then checked to resolve to that digest.
harbor_add_tag "${release_digest}" "${release_tag}" \
    "${artifact_directory}/release-manifest.json"

echo "Published signed immutable container ${release_digest} as ${release_reference}"
