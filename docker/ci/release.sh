#!/bin/sh
# Shared release-tag validation and checkout functions. Source this file from a
# script that has already enabled `set -eu`.

validate_release_tag() {
    if [ "$#" -ne 1 ] || ! printf '%s\n' "$1" \
        | grep -Eq '^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$'; then
        # Pre-release and build suffixes are rejected: Python normalizes them
        # (1.2.3-rc.1 becomes 1.2.3rc1), so the image label, wheel version, and
        # `--version` output would no longer match the tag.
        echo "release tag must be an unprefixed final Semantic Version such as 1.2.3" >&2
        return 2
    fi
}

checkout_release_tag() {
    if [ "$#" -ne 1 ]; then
        echo "usage: checkout_release_tag SEMVER" >&2
        return 2
    fi
    release_tag=$1
    validate_release_tag "${release_tag}"
    : "${CI_DEFAULT_BRANCH:?CI_DEFAULT_BRANCH is required}"

    git fetch --no-tags origin \
        "+refs/heads/${CI_DEFAULT_BRANCH}:refs/remotes/origin/${CI_DEFAULT_BRANCH}"
    remote_tag=$(git ls-remote --tags --refs origin "refs/tags/${release_tag}" \
        | awk 'NF {print $1; exit}')
    if [ -z "${remote_tag}" ]; then
        echo "release tag does not exist on origin: ${release_tag}" >&2
        return 1
    fi
    git fetch --no-tags origin \
        "refs/tags/${release_tag}:refs/tags/${release_tag}"
    release_tag_commit=$(git rev-parse --verify \
        "refs/tags/${release_tag}^{commit}")
    if ! git merge-base --is-ancestor \
        "${release_tag_commit}" "origin/${CI_DEFAULT_BRANCH}"; then
        echo "release tag is not on ${CI_DEFAULT_BRANCH}: ${release_tag}" >&2
        return 1
    fi
    git checkout --detach "${release_tag_commit}"
}
