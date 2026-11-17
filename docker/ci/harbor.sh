#!/bin/sh
# Shared Harbor registry and staging-tag functions. Source this file from a
# script that has already enabled `set -eu`.
#
# These functions use only the OCI distribution API (`/v2/`), with a bearer
# token from the registry's token service. Harbor's REST API (`/api/v2.0/`) is
# not reachable from the CI runners, so nothing here may depend on it.

# Every manifest type a release tag could point to, so a HEAD request finds an
# existing tag whatever it references.
harbor_manifest_accept='application/vnd.oci.image.manifest.v1+json, application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.v2+json, application/vnd.docker.distribution.manifest.list.v2+json'

# Every request is small (a challenge, a token, or a manifest), so a stalled
# connection fails the job instead of hanging it.
harbor_curl_connect_timeout=10
harbor_curl_max_time=120

# Run curl with the options every registry request uses. Callers capture its
# output with `|| { harbor_curl_failed "$?" ACTION; return 1; }`, so a failed
# request is reported as a curl failure, not as an HTTP status of 000.
harbor_curl() {
    curl --silent --show-error \
        --connect-timeout "${harbor_curl_connect_timeout}" \
        --max-time "${harbor_curl_max_time}" \
        "$@"
}

# Run harbor_curl with one secret option (`user` or `header`), given to curl
# as a config file on stdin, so the secret is not in curl's arguments, where
# other processes on the runner could read it. Usage:
#   harbor_curl_with_secret OPTION VALUE [CURL_ARGS...]
harbor_curl_with_secret() {
    harbor_secret_option=$1
    # In a curl config file, a quoted value needs backslashes and quotes escaped.
    harbor_secret_value=$(printf '%s' "$2" | sed 's/[\\"]/\\&/g')
    shift 2
    printf '%s = "%s"\n' "${harbor_secret_option}" "${harbor_secret_value}" | harbor_curl --config - "$@"
}

harbor_curl_failed() {
    echo "curl failed with exit status $1 while $2" >&2
}

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
    harbor_registry_repository="${HARBOR_PROJECT}/${HARBOR_REPO}/${harbor_repository_variant}"
    harbor_registry_url="https://${HARBOR_URL}/v2/${harbor_registry_repository}"
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

# Print the sha256 digest of REFERENCE@DIGEST or DIGEST, rejecting anything else.
harbor_digest() {
    if [ "$#" -ne 1 ]; then
        echo "usage: harbor_digest REFERENCE_OR_DIGEST" >&2
        return 2
    fi
    harbor_digest_value=${1#*@}
    if ! printf '%s\n' "${harbor_digest_value}" | grep -Eq '^sha256:[0-9a-f]{64}$'; then
        echo "invalid sha256 digest: $1" >&2
        return 2
    fi
    printf '%s\n' "${harbor_digest_value}"
}

# Print a registry bearer token for this repository with ACTIONS (such as
# `pull` or `pull,push`). The token service is the one named by the registry's
# `/v2/` challenge, and must be on the registry's own host, so the credentials
# are never sent anywhere else.
harbor_registry_token() {
    if [ "$#" -ne 1 ]; then
        echo "usage: harbor_registry_token ACTIONS" >&2
        return 2
    fi
    harbor_token_actions=$1
    harbor_challenge_response=$(harbor_curl \
        --output /dev/null \
        --write-out '%{http_code}\n%header{www-authenticate}' \
        "https://${HARBOR_URL}/v2/") || {
        harbor_curl_failed "$?" "requesting the registry's authentication challenge"
        return 1
    }
    harbor_challenge_status=$(printf '%s\n' "${harbor_challenge_response}" | sed -n '1p')
    harbor_challenge=$(printf '%s\n' "${harbor_challenge_response}" | sed -n '2p' | tr -d '\r')
    if [ "${harbor_challenge_status}" != "401" ]; then
        echo "registry returned HTTP ${harbor_challenge_status} for its authentication challenge" >&2
        return 1
    fi
    case "${harbor_challenge}" in
        [Bb]earer\ *) ;;
        *)
            echo "registry did not ask for a bearer token: ${harbor_challenge}" >&2
            return 1
            ;;
    esac
    harbor_token_realm=$(printf '%s\n' "${harbor_challenge}" | sed -n 's/.*realm="\([^"]*\)".*/\1/p')
    harbor_token_service=$(printf '%s\n' "${harbor_challenge}" | sed -n 's/.*service="\([^"]*\)".*/\1/p')
    case "${harbor_token_realm}" in
        "https://${HARBOR_URL}/"*) ;;
        *)
            echo "registry token service is not on ${HARBOR_URL}: ${harbor_token_realm}" >&2
            return 1
            ;;
    esac
    if [ -z "${harbor_token_service}" ]; then
        echo "registry challenge names no token service: ${harbor_challenge}" >&2
        return 1
    fi

    harbor_token_response=$(harbor_curl_with_secret \
        user "${OPENTEAMS_HARBOR_USERNAME}:${OPENTEAMS_HARBOR_TOKEN}" \
        --get \
        --data-urlencode "service=${harbor_token_service}" \
        --data-urlencode "scope=repository:${harbor_registry_repository}:${harbor_token_actions}" \
        --write-out '\n%{http_code}' \
        "${harbor_token_realm}") || {
        harbor_curl_failed "$?" "requesting a registry token"
        return 1
    }
    harbor_token_status=$(printf '%s\n' "${harbor_token_response}" | sed -n '$p')
    if [ "${harbor_token_status}" != "200" ]; then
        echo "registry token service returned HTTP ${harbor_token_status}" >&2
        return 1
    fi
    # Registry tokens are JWTs, which contain no quotes or backslashes.
    harbor_token_value=$(printf '%s\n' "${harbor_token_response}" \
        | sed -n 's/.*"token"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | sed -n '1p')
    if [ -z "${harbor_token_value}" ]; then
        harbor_token_value=$(printf '%s\n' "${harbor_token_response}" \
            | sed -n 's/.*"access_token"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | sed -n '1p')
    fi
    if [ -z "${harbor_token_value}" ]; then
        echo "registry token service returned no token" >&2
        return 1
    fi
    printf '%s\n' "${harbor_token_value}"
}

# Print "<HTTP status> <Docker-Content-Digest>" for a HEAD request on
# manifests/REFERENCE, a tag or digest.
harbor_head_manifest() {
    if [ "$#" -ne 2 ]; then
        echo "usage: harbor_head_manifest TOKEN REFERENCE" >&2
        return 2
    fi
    harbor_head_response=$(harbor_curl_with_secret \
        header "Authorization: Bearer $1" \
        --head \
        --header "Accept: ${harbor_manifest_accept}" \
        --output /dev/null \
        --write-out '%{http_code} %header{docker-content-digest}' \
        "${harbor_registry_url}/manifests/$2") || {
        harbor_curl_failed "$?" "checking ${harbor_image_base}:$2"
        return 1
    }
    printf '%s\n' "${harbor_head_response}" | tr -d '\r'
}

harbor_assert_release_tag_available() {
    if [ "$#" -ne 1 ]; then
        echo "usage: harbor_assert_release_tag_available SEMVER" >&2
        return 2
    fi
    harbor_release_tag=$1
    harbor_validate_tag "${harbor_release_tag}" || return 1
    harbor_release_reference="${harbor_image_base}:${harbor_release_tag}"
    harbor_pull_token=$(harbor_registry_token pull) || return 1
    harbor_head_result=$(harbor_head_manifest "${harbor_pull_token}" "${harbor_release_tag}") || return 1
    harbor_release_tag_status=${harbor_head_result%% *}
    case "${harbor_release_tag_status}" in
        404) ;;
        200)
            echo "immutable release tag already exists: ${harbor_release_reference}" >&2
            return 1
            ;;
        *)
            echo "registry returned HTTP ${harbor_release_tag_status} while checking ${harbor_release_reference}" >&2
            return 1
            ;;
    esac
}

# Tag the manifest DIGEST as TAG by pushing its exact bytes under TAG, keeping
# the digest unchanged. The fetched manifest is kept in MANIFEST_FILE. Fails
# unless TAG then resolves to DIGEST.
harbor_add_tag() {
    if [ "$#" -ne 3 ]; then
        echo "usage: harbor_add_tag DIGEST TAG MANIFEST_FILE" >&2
        return 2
    fi
    harbor_tag_digest=$(harbor_digest "$1") || return 1
    harbor_new_tag=$2
    harbor_manifest_file=$3
    harbor_validate_tag "${harbor_new_tag}" || return 1
    harbor_tag_reference="${harbor_image_base}:${harbor_new_tag}"
    harbor_push_token=$(harbor_registry_token pull,push) || return 1

    harbor_get_result=$(harbor_curl_with_secret \
        header "Authorization: Bearer ${harbor_push_token}" \
        --header "Accept: ${harbor_manifest_accept}" \
        --output "${harbor_manifest_file}" \
        --write-out '%{http_code} %{content_type}' \
        "${harbor_registry_url}/manifests/${harbor_tag_digest}") || {
        harbor_curl_failed "$?" "reading ${harbor_image_base}@${harbor_tag_digest}"
        return 1
    }
    harbor_get_status=${harbor_get_result%% *}
    harbor_manifest_media_type=${harbor_get_result#* }
    harbor_manifest_media_type=${harbor_manifest_media_type%%;*}
    if [ "${harbor_get_status}" != "200" ]; then
        echo "registry returned HTTP ${harbor_get_status} while reading ${harbor_image_base}@${harbor_tag_digest}" >&2
        return 1
    fi
    case "${harbor_manifest_media_type}" in
        application/vnd.oci.image.manifest.v1+json) ;;
        *)
            echo "${harbor_image_base}@${harbor_tag_digest} has media type ${harbor_manifest_media_type}, not an OCI image manifest" >&2
            return 1
            ;;
    esac
    harbor_fetched_sha256=$(sha256sum "${harbor_manifest_file}") || return 1
    harbor_fetched_digest="sha256:${harbor_fetched_sha256%% *}"
    if [ "${harbor_fetched_digest}" != "${harbor_tag_digest}" ]; then
        echo "manifest read for ${harbor_tag_digest} has digest ${harbor_fetched_digest}" >&2
        return 1
    fi

    harbor_put_response=$(harbor_curl_with_secret \
        header "Authorization: Bearer ${harbor_push_token}" \
        --request PUT \
        --header "Content-Type: ${harbor_manifest_media_type}" \
        --data-binary "@${harbor_manifest_file}" \
        --output /dev/null \
        --write-out '%{http_code} %header{docker-content-digest}' \
        "${harbor_registry_url}/manifests/${harbor_new_tag}") || {
        harbor_curl_failed "$?" "applying ${harbor_tag_reference}"
        return 1
    }
    harbor_put_result=$(printf '%s\n' "${harbor_put_response}" | tr -d '\r')
    harbor_put_status=${harbor_put_result%% *}
    harbor_put_digest=${harbor_put_result#* }
    if [ "${harbor_put_status}" != "201" ]; then
        echo "registry returned HTTP ${harbor_put_status} while applying ${harbor_tag_reference}" >&2
        return 1
    fi
    if [ -n "${harbor_put_digest}" ] && [ "${harbor_put_digest}" != "${harbor_tag_digest}" ]; then
        echo "registry stored ${harbor_tag_reference} as ${harbor_put_digest}, not ${harbor_tag_digest}" >&2
        return 1
    fi

    harbor_head_result=$(harbor_head_manifest "${harbor_push_token}" "${harbor_new_tag}") || return 1
    harbor_head_status=${harbor_head_result%% *}
    harbor_head_digest=${harbor_head_result#* }
    if [ "${harbor_head_status}" != "200" ] || [ "${harbor_head_digest}" != "${harbor_tag_digest}" ]; then
        echo "${harbor_tag_reference} resolves to '${harbor_head_digest}' (HTTP ${harbor_head_status}), not ${harbor_tag_digest}" >&2
        return 1
    fi
}
