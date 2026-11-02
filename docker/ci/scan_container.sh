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
syft_sbom="${artifact_directory}/sbom.syft.json"
grype_report="${artifact_directory}/grype-vulnerability-report.json"
trivy_template=${CHECKMAITE_TRIVY_TEMPLATE:-/usr/local/share/trivy/gitlab.tpl}
exceptions_file=${CHECKMAITE_DSOR3_EXCEPTIONS:-docker/ci/dsor3-exceptions.trivyignore.yaml}
# Medium findings and Grype findings have no available fix yet (unfixed Ubuntu
# CVEs; ffmpeg bundled in opencv and av, fixed only in ffmpeg 9), so the
# Medium-and-above Trivy gate and the Grype gate report without blocking until
# they are remediated or excepted (#799). Set to "true" to make them block.
strict_gates=${CHECKMAITE_STRICT_SCAN_GATES:-false}

case "${mode}" in
    enforce) exit_code=1 ;;
    report) exit_code=0 ;;
    *)
        echo "scan mode must be 'enforce' or 'report'" >&2
        exit 2
        ;;
esac

mkdir -p "${artifact_directory}"

# SPDX is retained for audit and attached to release digests by Cosign. The
# Syft JSON from the same catalogue is what Grype scans, as the program's
# cs-compliance component does.
syft "docker:${image}" \
    --output "spdx-json=${artifact_directory}/sbom.spdx.json" \
    --output "syft-json=${syft_sbom}"

# Scan once with each scanner, without severity or fix-availability filters. The
# retained JSON is the complete audit record; all other reports and the release
# gate are derived from that same result so they cannot disagree.
trivy image "${image}" \
    --scanners vuln \
    --format json \
    --output "${raw_report}" \
    --exit-code 0

# CS-2-H-2: Grype is the program's container scanner.
GRYPE_CHECK_FOR_APP_UPDATE=false grype "sbom:${syft_sbom}" \
    --output "json=${grype_report}" \
    --output "table=${artifact_directory}/grype-vulnerability-report.txt"

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

# The blocking gate fails on Trivy High and Critical findings, unfixed included.
# DSOR-3-H-2 (required from Maturity I) also permits no Medium findings, and
# CS-2-H-2 uses Grype: those two gates report, and block only when
# strict_gates is "true". The only findings excluded are unexpired entries in
# the exceptions file, which must cite an approved program exception, and only
# from the Trivy gates.
allowed_statements='Approved exception jatic/internal-docs#[0-9]+'
if [ "${mode}" = "enforce" ] && grep -E '^[[:space:]]*statement:' "${exceptions_file}" \
    | grep -Evq "^[[:space:]]*statement:[[:space:]]*\"?(${allowed_statements})"; then
    echo "${exceptions_file}: every statement must start with an approved exception ID" >&2
    exit 1
fi

if [ "${strict_gates}" = "true" ]; then
    strict_exit_code=${exit_code}
else
    strict_exit_code=0
fi

# Each gate table lists any findings that fail it (the Trivy tables also list
# the findings they accepted), in the job log and as an artifact. All gates run
# before the job fails so one log shows every finding.
gate_report="${artifact_directory}/release-gate.txt"
gate_status=0
trivy convert "${raw_report}" \
    --format table \
    --severity HIGH,CRITICAL \
    --ignorefile "${exceptions_file}" \
    --show-suppressed \
    --exit-code "${exit_code}" \
    --output "${gate_report}" || gate_status=$?
echo "Trivy gate (High, Critical; blocking):"
cat "${gate_report}"

medium_gate_report="${artifact_directory}/medium-release-gate.txt"
trivy convert "${raw_report}" \
    --format table \
    --severity MEDIUM,HIGH,CRITICAL \
    --ignorefile "${exceptions_file}" \
    --show-suppressed \
    --exit-code "${strict_exit_code}" \
    --output "${medium_gate_report}" || gate_status=$?
echo "Trivy gate (Medium, High, Critical; blocking only if CHECKMAITE_STRICT_SCAN_GATES=true):"
cat "${medium_gate_report}"

grype_gate_report="${artifact_directory}/grype-release-gate.txt"
jq -r '
    [.matches[]
        | select(.vulnerability.severity | IN("Medium", "High", "Critical"))
        | [.vulnerability.severity, .vulnerability.id, .artifact.name,
           .artifact.version, (.vulnerability.fix.state // "unknown"),
           ((.vulnerability.fix.versions // []) | join(","))]
    ]
    | unique
    | (["SEVERITY", "ID", "PACKAGE", "INSTALLED", "FIX STATE", "FIXED IN"], .[])
    | @tsv
' "${grype_report}" > "${grype_gate_report}"
grype_findings=$(($(wc -l < "${grype_gate_report}") - 1))
echo "Grype gate (Medium, High, Critical; blocking only if CHECKMAITE_STRICT_SCAN_GATES=true): ${grype_findings} finding(s)"
if [ "${grype_findings}" -gt 0 ]; then
    cat "${grype_gate_report}"
    if [ "${mode}" = "enforce" ] && [ "${strict_gates}" = "true" ]; then
        gate_status=1
    fi
fi
exit "${gate_status}"
