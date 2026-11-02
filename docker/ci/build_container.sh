#!/bin/sh
# Build one product container with the job's BuildKit builder. CI calls this
# once to load the image into the Docker daemon for testing and scanning and, in
# release jobs, publish_container.sh calls it again with the same arguments to
# push that fully cached result to Harbor with OCI media types (CS-1-H-1).
#
# Provenance attestations are disabled so the pushed artifact is a single OCI
# image manifest whose config digest can be compared with the scanned image;
# the SBOM and vulnerability report are attached to that digest by Cosign.
set -eu

if [ "$#" -lt 5 ]; then
    echo "usage: $0 cpu|cuda VERSION REVISION METADATA_FILE OUTPUT_ARGUMENT..." >&2
    exit 2
fi

target=$1
version=$2
revision=$3
metadata_file=$4
shift 4

case "${target}" in
    cpu|cuda) ;;
    *)
        echo "unsupported build target: ${target}" >&2
        exit 2
        ;;
esac

mkdir -p "$(dirname -- "${metadata_file}")"
docker buildx build \
    --platform "${TARGET_PLATFORM:-linux/amd64}" \
    --target "${target}" \
    --build-arg "CHECKMAITE_VERSION=${version}" \
    --label "org.opencontainers.image.revision=${revision}" \
    --provenance=false \
    --sbom=false \
    --metadata-file "${metadata_file}" \
    "$@" \
    .
