#!/bin/sh
set -eu

if [ "$#" -ne 3 ]; then
    echo "usage: $0 IMAGE VARIANT enforce|report" >&2
    exit 2
fi

image=$1
variant=$2
mode=$3
artifact_directory="artifacts/container/${variant}"
raw_report="${artifact_directory}/vulnerability-report.json"
trivy_template=${CHECKMAITE_TRIVY_TEMPLATE:-/usr/local/share/trivy/gitlab.tpl}
exceptions_file=${CHECKMAITE_DSOR3_EXCEPTIONS:-docker/ci/dsor3-exceptions.trivyignore.yaml}

case "${mode}" in
    enforce) exit_code=1 ;;
    report) exit_code=0 ;;
    *)
        echo "scan mode must be 'enforce' or 'report'" >&2
        exit 2
        ;;
esac

mkdir -p "${artifact_directory}"

# SPDX is retained for audit and attached to release digests by Cosign.
syft "docker:${image}" \
    --output "spdx-json=${artifact_directory}/sbom.spdx.json"

# Scan once without severity or fix-availability filters. The retained JSON is
# the complete audit record; all other reports and the release gate are derived
# from that same result so they cannot disagree.
trivy image "${image}" \
    --scanners vuln \
    --format json \
    --output "${raw_report}" \
    --exit-code 0

trivy convert "${raw_report}" \
    --format template \
    --template "@${trivy_template}" \
    --output "${artifact_directory}/gl-container-scanning-report.json"

trivy convert "${raw_report}" \
    --format table \
    --severity MEDIUM,HIGH,CRITICAL \
    --output "${artifact_directory}/vulnerability-report.txt"

# Cosign's vuln predicate format is attached to the published digest.
trivy convert "${raw_report}" \
    --format cosign-vuln \
    --output "${artifact_directory}/vulnerability-attestation.json"

# DSOR-3-H-2 permits no Medium, High, or Critical findings. Unfixed findings
# are included. The only findings excluded are those with an approved, unexpired
# program exception recorded in the exceptions file, and only from this gate.
if [ "${mode}" = "enforce" ] && grep -Eq '^[[:space:]]*statement:.*PENDING' "${exceptions_file}"; then
    echo "${exceptions_file} has exceptions without an approved DSOR-3 exception ID" >&2
    exit 1
fi
trivy convert "${raw_report}" \
    --format table \
    --severity MEDIUM,HIGH,CRITICAL \
    --ignorefile "${exceptions_file}" \
    --exit-code "${exit_code}" \
    --output /dev/null
