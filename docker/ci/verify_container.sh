#!/bin/sh
set -eu

if [ "$#" -ne 2 ]; then
    echo "usage: $0 IMAGE cpu|cuda" >&2
    exit 2
fi

image=$1
variant=$2
platform=${CONTAINER_TEST_PLATFORM:-linux/amd64}

case "${variant}" in
    cpu|cuda) ;;
    *)
        echo "variant must be 'cpu' or 'cuda'" >&2
        exit 2
        ;;
esac

docker run --rm --platform "${platform}" --user 0 --entrypoint /bin/bash "${image}" -lc '
    set -eu
    # `set -e` ignores commands negated with `!`, so fail explicitly.
    for name in dnf microdnf yum rpm rpmdb apt apt-get dpkg gpg gpgv tar pip pip3 gcc g++ cc c++ make git; do
        if command -v "${name}" >/dev/null 2>&1; then
            echo "unexpected executable in image: ${name}" >&2
            exit 1
        fi
    done
    /opt/venv/bin/python -c "import importlib.util; assert importlib.util.find_spec(\"pip\") is None; assert importlib.util.find_spec(\"ensurepip\") is None"
    test -z "$(find / -xdev -type f \( -perm -4000 -o -perm -2000 \) -print -quit)"
    # Package metadata and licence notices stay readable by scanners.
    test -s /var/lib/dpkg/status
    test -n "$(find /usr/share/doc -name copyright -print -quit)"
    test ! -e /var/lib/rpm && test ! -e /usr/lib/sysimage/rpm
    test ! -e /opt/venv/lib/python3.12/site-packages/ray/jars
'

docker run --rm --platform "${platform}" --entrypoint /bin/bash "${image}" -lc '
    set -eu
    test "$(id -u):$(id -g)" = "10001:10001"
    test ! -w /opt/venv
    test ! -w /checkmaite
    test -w /output
    test -w /cache
'

docker run --rm --platform "${platform}" --network none --read-only \
    --cap-drop ALL \
    --security-opt no-new-privileges \
    --tmpfs /output:rw,nosuid,nodev,noexec,mode=1777 \
    --tmpfs /cache:rw,nosuid,nodev,noexec,mode=1777 \
    --env "CHECKMAITE_CONTAINER_VARIANT=${variant}" \
    --interactive \
    --entrypoint /opt/venv/bin/python \
    "${image}" - < docker/ci/smoke_container.py
