#!/bin/sh
set -eu
# Without pipefail, a `docker export` that dies partway would leave `wc` with a
# short count that passes the budget check. CI runs this under BusyBox ash (the
# docker:cli image), which supports pipefail.
# shellcheck disable=SC3040
set -o pipefail

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
# Count the bytes of the image's exported filesystem. `docker image inspect`
# .Size also counts compressed layer content under the containerd image store
# (the Kubernetes runners' daemon), so it is not comparable across daemons.
container=$(docker create "${image}")
remove_container() {
    docker rm "${container}" >/dev/null 2>&1 || true
}
trap remove_container EXIT
trap 'exit 1' HUP INT TERM
actual=$(docker export "${container}" | wc -c | tr -d ' ')

mkdir -p "artifacts/container/${variant}"
printf '{\n  "actual_bytes": %s,\n  "maximum_bytes": %s\n}\n' \
    "${actual}" "${maximum}" \
    > "artifacts/container/${variant}/size-budget.json"

if [ "${actual}" -gt "${maximum}" ]; then
    echo "${variant} size ${actual} exceeds budget ${maximum}" >&2
    exit 1
fi
echo "${variant} size ${actual} is within budget ${maximum}"
