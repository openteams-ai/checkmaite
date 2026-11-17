"""Tests of docker/ci/harbor.sh and publish_container.sh against a fake registry.

A stub `curl` on PATH implements the parts of the OCI distribution API and
Harbor's token service that the release scripts use, so the scripts run
unmodified without network access or credentials.
"""

import hashlib
import json
import os
import shlex
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parents[2]
_CI_DIR = _REPO_ROOT / "docker/ci"
_HOST = "harbor.example.test"
_REPOSITORY = "openteams/checkmaite/cpu"
_USER = "robot$checkmaite"
_PASSWORD = "secret-token"  # noqa: S105
_OCI_MANIFEST = "application/vnd.oci.image.manifest.v1+json"

_SH = shutil.which("sh")
pytestmark = pytest.mark.skipif(_SH is None, reason="needs a POSIX shell")

_STUB_CURL = r"""
import hashlib, json, os, re, sys
from urllib.parse import parse_qs, urlencode

# Only the options harbor.sh passes are accepted, so a typo or a new option in
# the scripts fails the tests instead of being silently ignored.
VALUED = {
    "--output", "--write-out", "--header", "--user", "--data-urlencode", "--request", "--data-binary",
    "--connect-timeout", "--max-time", "--config",
}
# Options a config file on stdin may set, as harbor_curl_with_secret writes them.
CONFIG_KEYS = {"user", "header"}
CONFIG_LINE = re.compile(r'^([a-z-]+) = "((?:[^"\\]|\\.)*)"$')
FLAGS = {"--silent", "--show-error", "--head", "--get"}


def usage_error(message):
    sys.stderr.write(f"stub curl: {message}\n")
    sys.exit(2)


args = sys.argv[1:]
argv = list(args)
opts = {"--header": [], "--data-urlencode": []}
url = None
method = "GET"
use_get = False
i = 0
while i < len(args):
    a = args[i]
    if a in VALUED:
        if i + 1 >= len(args):
            usage_error(f"option {a} requires a value")
        v = args[i + 1]
        if a == "--config":
            if v != "-":
                usage_error("--config is only read from stdin")
            config = []
            for line in sys.stdin.read().splitlines():
                m = CONFIG_LINE.match(line)
                if not m or m.group(1) not in CONFIG_KEYS:
                    usage_error(f"unsupported config line {line!r}")
                config += [f"--{m.group(1)}", re.sub(r"\\(.)", r"\1", m.group(2))]
            args[i + 2 : i + 2] = config
            i += 2
            continue
        if a in ("--header", "--data-urlencode"):
            opts[a].append(v)
        elif a in opts:
            usage_error(f"option {a} given more than once")
        else:
            opts[a] = v
        i += 2
        continue
    if a == "--head":
        method = "HEAD"
    elif a == "--get":
        use_get = True
    elif a in FLAGS:
        pass
    elif a.startswith("-"):
        usage_error(f"unsupported option {a}")
    elif url is not None:
        usage_error(f"more than one URL: {url} {a}")
    else:
        url = a
    i += 1
if url is None:
    usage_error("no URL")

state_path = os.environ["FAKE_REGISTRY_STATE"]
with open(state_path) as f:
    state = json.load(f)

method = opts.get("--request", method)
headers_in = {}
for h in opts["--header"]:
    k, _, v = h.partition(":")
    headers_in[k.strip().lower()] = v.strip()
if use_get and opts["--data-urlencode"]:
    url = url + "?" + urlencode([tuple(d.split("=", 1)) for d in opts["--data-urlencode"]])

entry = {"method": method, "url": url, "argv": argv, "user": opts.get("--user"),
         "authorization": headers_in.get("authorization"),
         "content_type": headers_in.get("content-type"),
         "connect_timeout": opts.get("--connect-timeout"), "max_time": opts.get("--max-time")}
status, headers, body = 500, {}, b""
overrides = state.get("overrides", {})
host = state["host"]
key = f"{method} {url.split('?')[0]}"
# Simulate a transport failure: real curl still prints its --write-out, with
# an HTTP status of 000, and exits non-zero.
curl_exit = state.get("curl_exit", {}).get(key)
if curl_exit is not None:
    entry["curl_exit"] = curl_exit
    state.setdefault("log", []).append(entry)
    with open(state_path, "w") as f:
        json.dump(state, f)
    sys.stderr.write(f"curl: ({curl_exit}) simulated failure\n")
    fmt = opts.get("--write-out", "")
    sys.stdout.write(re.sub(r"%\{http_code\}", "000", re.sub(r"%header\{[\w-]+\}|%\{content_type\}", "", fmt)).replace("\\n", "\n"))
    sys.exit(curl_exit)
base = f"https://{host}/v2/"
if url == base:
    status = 401
    headers["WWW-Authenticate"] = state["challenge"]
elif url.startswith(state["realm"] + "?"):
    query = parse_qs(url.split("?", 1)[1])
    entry["scope"] = query.get("scope", [None])[0]
    if opts.get("--user") == state["credentials"]:
        status = 200
        granted = entry["scope"]
        if "granted_actions" in state:
            granted = granted.rsplit(":", 1)[0] + ":" + state["granted_actions"]
        body = json.dumps({"token": "tok." + granted.replace(":", ".").replace("/", ".")}).encode()
    else:
        status = 401
else:
    m = re.fullmatch(re.escape(base) + r"(.+)/manifests/([^/]+)", url)
    if not m or m.group(1) != state["repository"]:
        status = 404
    elif not (headers_in.get("authorization") or "").startswith("Bearer tok."):
        status = 401
    else:
        ref = m.group(2)
        manifests = state["manifests"]
        tags = state["tags"]
        if method in ("GET", "HEAD"):
            digest = ref if ref.startswith("sha256:") else tags.get(ref)
            if digest is not None and digest in manifests:
                status = 200
                stored = manifests[digest]
                body = stored["body"].encode()
                headers["Content-Type"] = stored["media_type"]
                headers["Docker-Content-Digest"] = state.get("served_digest", {}).get(ref, digest)
                if method == "HEAD":
                    body = b""
            else:
                status = 404
        elif method == "PUT" and "push" not in headers_in["authorization"].rsplit(".", 1)[1].split(","):
            status = 401
        elif method == "PUT":
            data_file = opts["--data-binary"]
            assert data_file.startswith("@")
            with open(data_file[1:], "rb") as f:
                data = f.read()
            digest = "sha256:" + hashlib.sha256(data).hexdigest()
            entry["put_digest"] = digest
            manifests.setdefault(digest, {"media_type": headers_in.get("content-type"), "body": data.decode()})
            if ref.startswith("sha256:"):
                status = 201
            elif ref in tags and ref in state.get("immutable", []):
                status = 412
            else:
                tags[ref] = digest
                status = 201
            headers["Docker-Content-Digest"] = digest
if key in overrides:
    status = overrides[key]
    if status != 200:
        headers.pop("Docker-Content-Digest", None)
state.setdefault("log", []).append(entry)
with open(state_path, "w") as f:
    json.dump(state, f)

out = opts.get("--output")
if out is not None and out != "/dev/null":
    with open(out, "wb") as f:
        f.write(body)
elif out is None and method != "HEAD":
    sys.stdout.buffer.write(body)

fmt = opts.get("--write-out", "")
lower = {k.lower(): v for k, v in headers.items()}
def sub(match):
    name = match.group(1) or match.group(2)
    if match.group(1) == "http_code":
        return f"{status:03d}"
    if match.group(1) == "content_type":
        return lower.get("content-type", "")
    if match.group(2):
        return lower.get(name.lower(), "")
    return match.group(0)
written = re.sub(r"%\{(\w+)\}|%header\{([\w-]+)\}", sub, fmt).replace("\\n", "\n")
sys.stdout.write(written)
"""

# publish_container.sh runs jq only as `jq -cS . FILE`; the test images have no jq.
_STUB_JQ = r"""
import json, sys

if len(sys.argv) != 4 or sys.argv[1:3] != ["-cS", "."]:
    sys.stderr.write(f"stub jq: unsupported arguments {sys.argv[1:]}\n")
    sys.exit(2)
with open(sys.argv[3]) as f:
    print(json.dumps(json.load(f), sort_keys=True, separators=(",", ":"), ensure_ascii=False))
"""

_STUB_COSIGN = r"""#!/bin/sh
printf '%s\n' "$*" >> "${COSIGN_LOG}"
if [ -n "${COSIGN_FAIL_ON:-}" ] && [ "$1" = "${COSIGN_FAIL_ON}" ]; then
    exit 1
fi
"""


def _manifest(config_byte: str = "a") -> str:
    return json.dumps(
        {
            "schemaVersion": 2,
            "mediaType": _OCI_MANIFEST,
            "config": {"mediaType": "application/vnd.oci.image.config.v1+json", "digest": "sha256:" + config_byte * 64},
            "layers": [],
        },
        indent=3,
    )


def _digest(body: str) -> str:
    return "sha256:" + hashlib.sha256(body.encode()).hexdigest()


class FakeRegistry:
    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.state_path = tmp_path / "registry.json"
        self.bin = tmp_path / "bin"
        self.bin.mkdir()
        curl = self.bin / "curl"
        curl.write_text(f"#!{sys.executable}\n{_STUB_CURL}")
        cosign = self.bin / "cosign"
        cosign.write_text(_STUB_COSIGN)
        jq = self.bin / "jq"
        jq.write_text(f"#!{sys.executable}\n{_STUB_JQ}")
        for tool in (curl, cosign, jq):
            tool.chmod(tool.stat().st_mode | stat.S_IXUSR)
        self.manifest = _manifest()
        self.digest = _digest(self.manifest)
        realm = f"https://{_HOST}/service/token"
        self.state = {
            "host": _HOST,
            "realm": realm,
            "challenge": f'Bearer realm="{realm}",service="harbor-registry"',
            "credentials": f"{_USER}:{_PASSWORD}",
            "repository": _REPOSITORY,
            "manifests": {self.digest: {"media_type": _OCI_MANIFEST, "body": self.manifest}},
            "tags": {"staging-1.2.3-42-cpu-amd64": self.digest},
            "overrides": {},
        }
        self.save()

    def save(self) -> None:
        self.state_path.write_text(json.dumps(self.state))

    def load(self) -> dict:
        return json.loads(self.state_path.read_text())

    def env(self) -> dict[str, str]:
        return {
            "PATH": f"{self.bin}{os.pathsep}{os.environ['PATH']}",
            "FAKE_REGISTRY_STATE": str(self.state_path),
            "COSIGN_LOG": str(self.tmp_path / "cosign.log"),
            "HARBOR_URL": _HOST,
            "HARBOR_PROJECT": "openteams",
            "HARBOR_REPO": "checkmaite",
            "OPENTEAMS_HARBOR_USERNAME": _USER,
            "OPENTEAMS_HARBOR_TOKEN": _PASSWORD,
            "CI_PIPELINE_ID": "42",
        }

    def run_functions(self, commands: str) -> subprocess.CompletedProcess[str]:
        script = f'set -eu\n. "{_CI_DIR / "harbor.sh"}"\nharbor_configure_repository cpu\n{commands}\n'
        return subprocess.run(  # noqa: S603
            [str(_SH), "-c", script], env=self.env(), cwd=self.tmp_path, capture_output=True, text=True, check=False
        )

    def requests(self, method: str) -> list[dict]:
        return [entry for entry in self.load().get("log", []) if entry["method"] == method]


@pytest.fixture
def registry(tmp_path: Path) -> FakeRegistry:
    return FakeRegistry(tmp_path)


def test_release_tag_available_uses_registry_api_with_pull_token(registry: FakeRegistry) -> None:
    result = registry.run_functions("harbor_assert_release_tag_available 1.2.3")

    assert result.returncode == 0, result.stderr
    log = registry.load()["log"]
    assert all("/api/" not in entry["url"] for entry in log)
    token_requests = [entry for entry in log if entry.get("scope")]
    assert [entry["scope"] for entry in token_requests] == [f"repository:{_REPOSITORY}:pull"]
    # Credentials go only to the token service named by the challenge.
    assert [entry["url"].split("?")[0] for entry in log if entry["user"]] == [registry.state["realm"]]
    heads = registry.requests("HEAD")
    assert [entry["url"] for entry in heads] == [f"https://{_HOST}/v2/{_REPOSITORY}/manifests/1.2.3"]


def test_every_request_has_timeouts(registry: FakeRegistry) -> None:
    manifest_file = registry.tmp_path / "manifest.json"

    result = registry.run_functions(
        f'harbor_assert_release_tag_available 1.2.3\nharbor_add_tag "{registry.digest}" 1.2.3 "{manifest_file}"'
    )

    assert result.returncode == 0, result.stderr
    log = registry.load()["log"]
    assert {entry["method"] for entry in log} == {"GET", "HEAD", "PUT"}
    assert all((entry["connect_timeout"], entry["max_time"]) == ("10", "120") for entry in log)


def test_secrets_are_not_curl_arguments(registry: FakeRegistry) -> None:
    """Credentials and tokens reach curl on stdin, never as arguments other processes could read."""
    manifest_file = registry.tmp_path / "manifest.json"

    result = registry.run_functions(
        f'harbor_assert_release_tag_available 1.2.3\nharbor_add_tag "{registry.digest}" 1.2.3 "{manifest_file}"'
    )

    assert result.returncode == 0, result.stderr
    log = registry.load()["log"]
    assert any(entry["user"] for entry in log)
    assert any(entry["authorization"] for entry in log)
    for entry in log:
        arguments = " ".join(entry["argv"])
        assert _PASSWORD not in arguments
        assert "Authorization" not in arguments
        assert "tok." not in arguments


def test_credentials_with_quotes_and_backslashes_reach_the_token_service(registry: FakeRegistry) -> None:
    password = 'p"a\\ss$word'  # noqa: S105
    registry.state["credentials"] = f"{_USER}:{password}"
    registry.save()

    result = registry.run_functions(
        f"OPENTEAMS_HARBOR_TOKEN={shlex.quote(password)}\nharbor_assert_release_tag_available 1.2.3"
    )

    assert result.returncode == 0, result.stderr
    assert [entry["user"] for entry in registry.load()["log"] if entry["user"]] == [f"{_USER}:{password}"]


def test_functions_leave_caller_variables_alone(registry: FakeRegistry) -> None:
    names = (
        "release_tag release_reference status head_result pull_token push_token new_tag manifest_file "
        "tag_digest tag_reference get_result put_result challenge token_realm registry_token"
    ).split()
    manifest_file = registry.tmp_path / "manifest.json"
    commands = "\n".join(
        [
            *(f"{name}=caller" for name in names),
            "harbor_assert_release_tag_available 1.2.3",
            f'harbor_add_tag "{registry.digest}" 1.2.3 "{manifest_file}"',
            *(f'printf "%s=%s\\n" {name} "${{{name}}}"' for name in names),
        ]
    )

    result = registry.run_functions(commands)

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [f"{name}=caller" for name in names]


def test_existing_release_tag_is_refused(registry: FakeRegistry) -> None:
    registry.state["tags"]["1.2.3"] = registry.digest
    registry.save()

    result = registry.run_functions("harbor_assert_release_tag_available 1.2.3")

    assert result.returncode == 1
    assert "immutable release tag already exists" in result.stderr


def test_unexpected_status_fails_closed(registry: FakeRegistry) -> None:
    registry.state["overrides"][f"HEAD https://{_HOST}/v2/{_REPOSITORY}/manifests/1.2.3"] = 403
    registry.save()

    result = registry.run_functions("harbor_assert_release_tag_available 1.2.3")

    assert result.returncode == 1
    assert "registry returned HTTP 403 while checking" in result.stderr


def test_bad_credentials_fail_closed(registry: FakeRegistry) -> None:
    registry.state["credentials"] = "someone:else"
    registry.save()

    result = registry.run_functions("harbor_assert_release_tag_available 1.2.3")

    assert result.returncode != 0
    assert "token service returned HTTP 401" in result.stderr
    assert registry.requests("HEAD") == []


def test_token_service_on_another_host_is_not_sent_credentials(registry: FakeRegistry) -> None:
    registry.state["challenge"] = 'Bearer realm="https://elsewhere.example.test/token",service="harbor-registry"'
    registry.save()

    result = registry.run_functions("harbor_assert_release_tag_available 1.2.3")

    assert result.returncode != 0
    assert "is not on" in result.stderr
    assert all(entry["user"] is None for entry in registry.load()["log"])


def test_add_tag_pushes_identical_manifest_and_verifies_digest(registry: FakeRegistry) -> None:
    manifest_file = registry.tmp_path / "release-manifest.json"

    result = registry.run_functions(f'harbor_add_tag "{_HOST}/{_REPOSITORY}@{registry.digest}" 1.2.3 "{manifest_file}"')

    assert result.returncode == 0, result.stderr
    state = registry.load()
    assert state["tags"]["1.2.3"] == registry.digest
    assert manifest_file.read_text() == registry.manifest
    (put,) = registry.requests("PUT")
    assert put["url"] == f"https://{_HOST}/v2/{_REPOSITORY}/manifests/1.2.3"
    assert put["content_type"] == _OCI_MANIFEST
    assert put["put_digest"] == registry.digest
    assert [entry["scope"] for entry in state["log"] if entry.get("scope")] == [f"repository:{_REPOSITORY}:pull,push"]
    # The tag is resolved after it is pushed.
    assert state["log"][-1]["method"] == "HEAD"
    assert state["log"][-1]["url"].endswith("/manifests/1.2.3")
    assert all("/api/" not in entry["url"] for entry in state["log"])


def test_add_tag_refuses_manifest_with_other_digest(registry: FakeRegistry) -> None:
    other = _manifest("b")
    registry.state["manifests"][registry.digest]["body"] = other
    registry.save()

    result = registry.run_functions(f'harbor_add_tag "{registry.digest}" 1.2.3 "{registry.tmp_path / "manifest.json"}"')

    assert result.returncode == 1
    assert f"has digest {_digest(other)}" in result.stderr
    assert registry.requests("PUT") == []


def test_add_tag_refuses_non_oci_manifest(registry: FakeRegistry) -> None:
    registry.state["manifests"][registry.digest]["media_type"] = "application/vnd.docker.distribution.manifest.v2+json"
    registry.save()

    result = registry.run_functions(f'harbor_add_tag "{registry.digest}" 1.2.3 "{registry.tmp_path / "manifest.json"}"')

    assert result.returncode == 1
    assert "not an OCI image manifest" in result.stderr
    assert registry.requests("PUT") == []


def test_add_tag_fails_when_tag_resolves_elsewhere(registry: FakeRegistry) -> None:
    registry.state["served_digest"] = {"1.2.3": "sha256:" + "c" * 64}
    registry.save()

    result = registry.run_functions(f'harbor_add_tag "{registry.digest}" 1.2.3 "{registry.tmp_path / "manifest.json"}"')

    assert result.returncode == 1
    assert "resolves to" in result.stderr


def test_add_tag_reports_rejected_push(registry: FakeRegistry) -> None:
    registry.state["tags"]["1.2.3"] = registry.digest
    registry.state["immutable"] = ["1.2.3"]
    registry.save()

    result = registry.run_functions(f'harbor_add_tag "{registry.digest}" 1.2.3 "{registry.tmp_path / "manifest.json"}"')

    assert result.returncode == 1
    assert "registry returned HTTP 412 while applying" in result.stderr


_MANIFESTS = f"https://{_HOST}/v2/{_REPOSITORY}/manifests"
_ASSERT = "harbor_assert_release_tag_available 1.2.3"


def _add_tag(registry: FakeRegistry) -> str:
    return f'harbor_add_tag "{registry.digest}" 1.2.3 "{registry.tmp_path / "manifest.json"}"'


@pytest.mark.parametrize(
    ("step", "request_key", "action"),
    [
        ("check", f"GET https://{_HOST}/v2/", "requesting the registry's authentication challenge"),
        ("check", f"GET https://{_HOST}/service/token", "requesting a registry token"),
        ("check", f"HEAD {_MANIFESTS}/1.2.3", "checking"),
        ("tag", "GET {manifests}/{digest}", "reading"),
        ("tag", f"PUT {_MANIFESTS}/1.2.3", "applying"),
        ("tag", f"HEAD {_MANIFESTS}/1.2.3", "checking"),
    ],
)
def test_curl_failure_fails_closed(registry: FakeRegistry, step: str, request_key: str, action: str) -> None:
    request_key = request_key.format(manifests=_MANIFESTS, digest=registry.digest)
    registry.state["curl_exit"] = {request_key: 28}
    registry.save()

    result = registry.run_functions(_ASSERT if step == "check" else _add_tag(registry))

    assert result.returncode == 1
    assert f"curl failed with exit status 28 while {action}" in result.stderr
    assert "HTTP 000" not in result.stderr
    assert [entry.get("curl_exit") for entry in registry.load()["log"]][-1] == 28


def test_add_tag_fails_closed_when_manifest_read_is_unauthorized(registry: FakeRegistry) -> None:
    registry.state["overrides"][f"GET {_MANIFESTS}/{registry.digest}"] = 401
    registry.save()

    result = registry.run_functions(_add_tag(registry))

    assert result.returncode == 1
    assert "registry returned HTTP 401 while reading" in result.stderr
    assert registry.requests("PUT") == []


def test_add_tag_fails_closed_without_push_permission(registry: FakeRegistry) -> None:
    registry.state["granted_actions"] = "pull"
    registry.save()

    result = registry.run_functions(_add_tag(registry))

    assert result.returncode == 1
    assert "registry returned HTTP 401 while applying" in result.stderr
    state = registry.load()
    assert [entry["scope"] for entry in state["log"] if entry.get("scope")] == [f"repository:{_REPOSITORY}:pull,push"]
    assert "1.2.3" not in state["tags"]


def _publish(registry: FakeRegistry, **extra_env: str) -> subprocess.CompletedProcess[str]:
    workdir = registry.tmp_path / "work"
    (workdir / "docker/ci").mkdir(parents=True, exist_ok=True)
    shutil.copy(_CI_DIR / "cosign-signing-config.json", workdir / "docker/ci/cosign-signing-config.json")
    artifacts = workdir / "artifacts/container/cpu-amd64"
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "sbom.spdx.json").write_text("{}")
    (artifacts / "vulnerability-attestation.json").write_text("{}")
    (artifacts / "staging-digest.txt").write_text(f"{_HOST}/{_REPOSITORY}@{registry.digest}\n")
    env = registry.env() | {"COSIGN_KMS_KEY": "alias/test"} | extra_env
    return subprocess.run(  # noqa: S603
        [str(_SH), str(_CI_DIR / "publish_container.sh"), "cpu", "cpu-amd64", "1.2.3", "release"],
        env=env,
        cwd=workdir,
        capture_output=True,
        text=True,
        check=False,
    )


def test_publish_applies_release_tag_after_verification(registry: FakeRegistry) -> None:
    result = _publish(registry)

    assert result.returncode == 0, result.stderr
    assert registry.load()["tags"]["1.2.3"] == registry.digest
    # The staging tag is kept: the registry API cannot remove only a tag.
    assert registry.load()["tags"]["staging-1.2.3-42-cpu-amd64"] == registry.digest
    cosign_calls = (registry.tmp_path / "cosign.log").read_text().splitlines()
    assert [call.split()[0] for call in cosign_calls] == [
        "sign",
        "attest",
        "attest",
        "verify",
        "verify-attestation",
        "verify-attestation",
    ]
    assert all(call.endswith(registry.digest) for call in cosign_calls)
    assert len(registry.requests("PUT")) == 1
    published = registry.tmp_path / "work/artifacts/container/cpu-amd64"
    assert (published / "published-digest.txt").read_text().strip().endswith(registry.digest)
    assert (published / "release-manifest.json").read_text() == registry.manifest


def test_publish_dry_run_applies_only_a_dry_run_tag(registry: FakeRegistry) -> None:
    result = _publish(registry, CONTAINER_RELEASE_DRY_RUN="true")

    assert result.returncode == 0, result.stderr
    assert "no release tag was applied" in result.stdout
    state = registry.load()
    assert "1.2.3" not in state["tags"]
    assert state["tags"]["staging-dryrun-42-cpu-amd64"] == registry.digest
    (put,) = registry.requests("PUT")
    assert put["url"] == f"{_MANIFESTS}/staging-dryrun-42-cpu-amd64"
    assert put["put_digest"] == registry.digest
    assert f"repository:{_REPOSITORY}:pull,push" in [entry.get("scope") for entry in state["log"]]
    assert state["log"][-1]["method"] == "HEAD"
    assert state["log"][-1]["url"] == f"{_MANIFESTS}/staging-dryrun-42-cpu-amd64"
    assert "verify-attestation" in (registry.tmp_path / "cosign.log").read_text()


def test_publish_dry_run_fails_when_tag_write_fails(registry: FakeRegistry) -> None:
    registry.state["granted_actions"] = "pull"
    registry.save()

    result = _publish(registry, CONTAINER_RELEASE_DRY_RUN="true")

    assert result.returncode == 1
    assert "registry returned HTTP 401 while applying" in result.stderr
    assert "no release tag was applied" not in result.stdout
    assert registry.load()["tags"].keys() == {"staging-1.2.3-42-cpu-amd64"}


@pytest.mark.parametrize("failing_step", ["sign", "attest", "verify", "verify-attestation"])
def test_publish_failure_before_tagging_applies_no_release_tag(registry: FakeRegistry, failing_step: str) -> None:
    result = _publish(registry, COSIGN_FAIL_ON=failing_step)

    assert result.returncode != 0
    assert "1.2.3" not in registry.load()["tags"]
    assert registry.requests("PUT") == []


def test_publish_refuses_existing_release_tag(registry: FakeRegistry) -> None:
    registry.state["tags"]["1.2.3"] = registry.digest
    registry.save()

    result = _publish(registry)

    assert result.returncode == 1
    assert "immutable release tag already exists" in result.stderr
    assert not (registry.tmp_path / "cosign.log").exists()


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--bogus", "https://example.test/"], "unsupported option --bogus"),
        (["-s", "https://example.test/"], "unsupported option -s"),
        (["https://example.test/", "--output"], "option --output requires a value"),
        (["https://example.test/a", "https://example.test/b"], "more than one URL"),
        (["--silent"], "no URL"),
    ],
)
def test_stub_curl_rejects_unknown_usage(registry: FakeRegistry, arguments: list[str], message: str) -> None:
    result = subprocess.run(  # noqa: S603
        [str(registry.bin / "curl"), *arguments],
        env=registry.env(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert f"stub curl: {message}" in result.stderr
    assert "log" not in registry.load()
