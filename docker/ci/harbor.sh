#!/bin/sh
# Shared Harbor API and staging-tag functions. Source this file from a script
# that has already enabled `set -eu`.

harbor_configure_repository() {
    if [ "$#" -ne 1 ]; then
        echo "usage: harbor_configure_repository VARIANT" >&2
        return 2
    fi
    harbor_repository_variant=$1
    case "${harbor_repository_variant}" in
        cpu|cuda) ;;
        *)
            echo "unsupported Harbor repository variant: ${harbor_repository_variant}" >&2
            return 2
            ;;
    esac

    : "${HARBOR_URL:?HARBOR_URL is required}"
    : "${HARBOR_PROJECT:?HARBOR_PROJECT is required}"
    : "${HARBOR_REPO:?HARBOR_REPO is required}"
    : "${OPENTEAMS_HARBOR_USERNAME:?OPENTEAMS_HARBOR_USERNAME is required}"
    : "${OPENTEAMS_HARBOR_TOKEN:?OPENTEAMS_HARBOR_TOKEN is required}"

    harbor_image_base="${HARBOR_URL}/${HARBOR_PROJECT}/${HARBOR_REPO}/${harbor_repository_variant}"
    harbor_repository_path="${HARBOR_REPO}%252F${harbor_repository_variant}"
    harbor_repository_url="https://${HARBOR_URL}/api/v2.0/projects/${HARBOR_PROJECT}/repositories/${harbor_repository_path}"
}

harbor_staging_tag() {
    if [ "$#" -ne 2 ]; then
        echo "usage: harbor_staging_tag SEMVER ARTIFACT_VARIANT" >&2
        return 2
    fi
    : "${CI_PIPELINE_ID:?CI_PIPELINE_ID is required}"
    printf 'staging-%s-%s-%s\n' "$1" "${CI_PIPELINE_ID}" "$2"
}

harbor_validate_tag() {
    if [ "$#" -ne 1 ] || [ -z "$1" ]; then
        echo "Harbor tag is empty" >&2
        return 2
    fi
    case "$1" in
        *[!0-9A-Za-z._-]*)
            echo "invalid Harbor tag: $1" >&2
            return 2
            ;;
    esac
}

harbor_encode_digest() {
    if [ "$#" -ne 1 ]; then
        echo "usage: harbor_encode_digest REFERENCE_OR_DIGEST" >&2
        return 2
    fi
    digest_value=${1#*@}
    if ! printf '%s\n' "${digest_value}" | grep -Eq '^sha256:[0-9a-f]{64}$'; then
        echo "invalid sha256 digest: $1" >&2
        return 2
    fi
    printf '%s%%3A%s\n' "${digest_value%%:*}" "${digest_value#*:}"
}

harbor_assert_release_tag_available() {
    if [ "$#" -ne 1 ]; then
        echo "usage: harbor_assert_release_tag_available SEMVER" >&2
        return 2
    fi
    release_tag=$1
    harbor_validate_tag "${release_tag}"
    release_reference="${harbor_image_base}:${release_tag}"
    status=$(curl --silent --show-error \
        --user "${OPENTEAMS_HARBOR_USERNAME}:${OPENTEAMS_HARBOR_TOKEN}" \
        --header 'Accept: application/json' \
        --output /dev/null \
        --write-out '%{http_code}' \
        "${harbor_repository_url}/artifacts/${release_tag}")
    case "${status}" in
        404) ;;
        200)
            echo "immutable release tag already exists: ${release_reference}" >&2
            return 1
            ;;
        *)
            echo "registry returned HTTP ${status} while checking ${release_reference}" >&2
            return 1
            ;;
    esac
}

harbor_remove_tag() {
    if [ "$#" -ne 2 ]; then
        echo "usage: harbor_remove_tag DIGEST TAG" >&2
        return 2
    fi
    encoded_digest=$(harbor_encode_digest "$1")
    harbor_validate_tag "$2"
    status=$(curl --silent --show-error \
        --user "${OPENTEAMS_HARBOR_USERNAME}:${OPENTEAMS_HARBOR_TOKEN}" \
        --request DELETE \
        --output /dev/null \
        --write-out '%{http_code}' \
        "${harbor_repository_url}/artifacts/${encoded_digest}/tags/$2")
    if [ "${status}" != "200" ]; then
        echo "registry returned HTTP ${status} while removing ${harbor_image_base}:$2" >&2
        return 1
    fi
}

harbor_add_tag() {
    if [ "$#" -ne 2 ]; then
        echo "usage: harbor_add_tag DIGEST TAG" >&2
        return 2
    fi
    encoded_digest=$(harbor_encode_digest "$1")
    harbor_validate_tag "$2"
    status=$(curl --silent --show-error \
        --user "${OPENTEAMS_HARBOR_USERNAME}:${OPENTEAMS_HARBOR_TOKEN}" \
        --request POST \
        --header 'Content-Type: application/json' \
        --data "{\"name\":\"$2\"}" \
        --output /dev/null \
        --write-out '%{http_code}' \
        "${harbor_repository_url}/artifacts/${encoded_digest}/tags")
    if [ "${status}" != "201" ]; then
        echo "registry returned HTTP ${status} while applying ${harbor_image_base}:$2" >&2
        return 1
    fi
}
