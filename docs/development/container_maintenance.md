# Maintain the product containers

The product containers use explicit pins for their base containers, operating
system packages, Python dependencies, build tools, and security tools. Update
these inputs together and retain the resulting scan and test evidence.

## Pinning policy

The standards do not require every operating-system package to have an exact
version. DSOR-3 requires vulnerability scanning, and CS-3 recommends
digest-pinned base containers. CheckMAITE additionally pins every package
installed after the base is selected, because a base digest does not pin later
downloads from a moving archive.

Both products are built on Ubuntu 24.04: the CPU product on the Docker Official
`ubuntu:24.04` image and the CUDA product on NVIDIA's CUDA 13 cuDNN runtime
image for Ubuntu 24.04, both digest-pinned. `docker/snapshot_apt.sh` replaces
every apt source with the Ubuntu archive snapshot named by `UBUNTU_SNAPSHOT` in
`Dockerfile`, upgrades every package to that snapshot, and installs only the
named packages. A rebuild therefore sees the same package versions until the
timestamp moves. Any index that fails to download fails the build. NVIDIA's apt
source is removed, so CUDA, cuDNN, and NCCL packages are never changed.

UBI is not used. At the time of the change, UBI 9 and UBI 10 carried HIGH
findings that Red Hat had not fixed (pcre2, util-linux, openssl, expat,
python3.12), while Ubuntu 24.04 had none at the pinned snapshot.

The plain Ubuntu image ships without a CA bundle, and snapshot.ubuntu.com is
HTTPS only. `docker/snapshot_apt.sh` therefore fetches `ca-certificates` once
without TLS peer verification. apt still verifies every index and package
against the archive signing key in the digest-pinned base, which is the trust
model of Ubuntu's default HTTP mirrors. Every later operation verifies TLS.

## Refresh base containers and the Ubuntu snapshot

1. Select supported `ubuntu:24.04`, NVIDIA CUDA cuDNN runtime for Ubuntu 24.04,
   and uv releases. Resolve each tag to a registry digest and update the three
   image `ARG` values in `Dockerfile`. Review the upstream release notes and
   verify that the bases support the required platforms.
2. Move `UBUNTU_SNAPSHOT` in `Dockerfile` to a current UTC timestamp in the
   form `YYYYMMDDTHHMMSSZ` (see <https://snapshot.ubuntu.com>). Record the old
   and new timestamps and the scan difference in the change description.
3. Build CPU and CUDA and compare the vulnerability reports with the previous
   ones. A snapshot never changes, so a failed build after a refresh points to
   a base, package, or network problem rather than to a removed version.

## Refresh Python locks

Update the root lock using the project's normal uv workflow. Then seed the
private container lock from it so shared dependencies retain the root project's
versions before uv resolves the device variants:

```bash
cp uv.lock docker/runtime/uv.lock
uv lock --directory docker/runtime --python 3.12
uv run python docker/ci/check_runtime_lock.py
```

Inspect the resulting device closures. CPU must contain CPU-only PyTorch and
`onnxruntime`; CUDA must contain CUDA PyTorch, `onnxruntime-gpu`, and Triton.
The private lock is explained in `docker/runtime/README.md`.

## Refresh CI and security-tool pins

Review and update these locations:

- Docker, Docker-in-Docker, and BuildKit image digests in
  `.gitlab/container.gitlab-ci.yml`.
- The Dockerfile frontend digest on the first line of `Dockerfile`.
- Syft, Trivy, and the Trivy GitLab template versions and SHA-256
  checksums in `docker/ci/install_security_tools.sh`.
- The digest-pinned uv container in `Dockerfile`.

Obtain checksums from the upstream release and verify downloaded bytes before
committing them. Keep the version and its checksum in the same change. Do not
replace digest or checksum verification with a mutable tag.

## Validate a refresh

Run the repository checks first:

```bash
uv run pre-commit run --all-files
uv run pyright docker/runtime/src/checkmaite_container docker/ci \
  tests/test_container
uv run pytest tests/test_container
uv run python docker/runtime/generate_schema.py --check
uv run python docker/ci/check_runtime_lock.py
shellcheck docker/*.sh docker/ci/*.sh
```

Then build and verify Linux AMD64 CPU and Linux AMD64 CUDA.
For each product, run `docker/ci/verify_container.sh`,
`docker/ci/check_container_size.sh`, and the normal SBOM and vulnerability scan.
Confirm that:

- CPU remains below its 3.00 GiB uncompressed limit.
- CUDA remains below its 11.00 GiB uncompressed limit.
- CPU contains no CUDA, NVIDIA, or Triton packages.
- CUDA retains NVIDIA's environment and driver constraints.
- Package managers (apt, dpkg), gpg, tar, build tools, pip, ensurepip, and
  setuid/setgid files are absent, while dpkg metadata and every package's
  `copyright` file remain available to scanners.
- Ray's bundled Java jar, Ray's vendored aiohttp, and virtualenv are absent
  from the virtual environment. `docker/install_locked_environment.sh` removes
  them after `uv pip check`, because the batch runtime cannot use them and they
  carry findings with no available upgrade. Recheck this after every Ray
  upgrade.
- The retained vulnerability JSON is unfiltered and the Medium, High, and
  Critical gate passes or has a formal program exception.

Physical CUDA execution and performance validation still require a supported
NVIDIA host.
