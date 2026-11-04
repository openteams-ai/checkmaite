#!/bin/sh
# Push a release candidate to a pipeline-scoped Harbor staging tag and record
# its digest. Signing needs the program AWS KMS key, which only the
# k8s-small-cpu runners can reach, so publish_container.sh signs and tags the
# digest in a separate job.
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
: "${CI_PIPELINE_ID:?CI_PIPELINE_ID is required}"

harbor_assert_release_tag_available "${release_tag}"
staging_tag=$(harbor_staging_tag "${release_tag}" "${artifact_variant}")
staging_reference="${harbor_image_base}:${staging_tag}"

artifact_directory="artifacts/container/${artifact_variant}"
build_metadata="${artifact_directory}/build-metadata.json"
push_metadata="${artifact_directory}/push-metadata.json"
test -s "${build_metadata}"

# The scanned image is the one the build step loaded into the daemon. Its config
# digest covers its configuration and every layer's uncompressed digest. The
# build step records it, whichever image store the daemon uses.
scanned_config=$(cat "${artifact_directory}/scanned-config.txt")
printf '%s\n' "${scanned_config}" | grep -Eq '^sha256:[0-9a-f]{64}$'
revision=$(docker image inspect \
    --format '{{index .Config.Labels "org.opencontainers.image.revision"}}' \
    "${local_image}")
test -n "${revision}"

# Push a pipeline-scoped staging tag so the digest exists in the registry for
# Cosign. The Semantic Version tag is applied only after verification, and
# publish_container.sh removes the staging tag.
# `docker push` from the daemon's image store re-encodes the image with Docker
# media types, which CS-1-H-1 rejects, so BuildKit pushes the same, fully cached
# build with OCI media types instead.
sh "${script_directory}/build_container.sh" \
    "${repository_variant}" "${release_tag}" "${revision}" "${push_metadata}" \
    --output "type=image,name=${staging_reference},push=true,oci-mediatypes=true"

pushed_digest=$(jq -r '."containerimage.digest" // empty' "${push_metadata}")
staging_digest="${harbor_image_base}@${pushed_digest}"
harbor_encode_digest "${staging_digest}" >/dev/null

# Remove the staging tag if a check below fails; on success the signing job
# (publish_container.sh) removes it.
staged=false
remove_failed_staging() {
    if [ "${staged}" != "true" ]; then
        harbor_remove_tag "${staging_digest}" "${staging_tag}" || true
    fi
}
trap remove_failed_staging EXIT
trap 'exit 1' HUP INT TERM

# Read the manifest back from Harbor: it must be an OCI image manifest for the
# exact image that was tested and scanned.
published_manifest="${artifact_directory}/published-manifest.json"
docker buildx imagetools inspect --raw "${staging_digest}" > "${published_manifest}"
published_media_type=$(jq -r '.mediaType' "${published_manifest}")
published_config=$(jq -r '.config.digest' "${published_manifest}")
if [ "${published_media_type}" != "application/vnd.oci.image.manifest.v1+json" ]; then
    echo "${staging_digest} has media type ${published_media_type}, not an OCI image manifest" >&2
    exit 1
fi
if [ "${published_config}" != "${scanned_config}" ]; then
    echo "pushed config ${published_config} does not match scanned image config ${scanned_config}" >&2
    exit 1
fi

printf '%s\n' "${staging_digest}" > "${artifact_directory}/staging-digest.txt"
staged=true
echo "Staged ${staging_digest} as ${staging_reference}"
