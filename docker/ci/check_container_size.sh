#!/bin/sh
set -eu

if [ "$#" -ne 2 ]; then
    echo "usage: $0 IMAGE cpu-amd64|cuda" >&2
    exit 2
fi

image=$1
variant=$2
case "${variant}" in
    cpu-amd64|cuda) ;;
    *)
        echo "unsupported size variant: ${variant}" >&2
        exit 2
        ;;
esac

maximum=$(jq -r \
    --arg variant "${variant}" \
    '.budgets[] | select(.variant == $variant) | .maximum_uncompressed_bytes' \
    docker/container-size-budgets.json)
test -n "${maximum}"
test "${maximum}" != "null"
actual=$(docker image inspect --format '{{.Size}}' "${image}")

mkdir -p "artifacts/container/${variant}"
printf '{\n  "actual_bytes": %s,\n  "maximum_bytes": %s\n}\n' \
    "${actual}" "${maximum}" \
    > "artifacts/container/${variant}/size-budget.json"

if [ "${actual}" -gt "${maximum}" ]; then
    echo "${variant} size ${actual} exceeds budget ${maximum}" >&2
    exit 1
fi
echo "${variant} size ${actual} is within budget ${maximum}"
