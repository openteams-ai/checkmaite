#!/bin/sh
set -eu

if [ "$#" -ne 1 ]; then
    echo "usage: $0 WHEEL_DIRECTORY" >&2
    exit 2
fi

wheel_directory=$1

test -d "${wheel_directory}"
UV_COMPILE_BYTECODE=1 uv pip install \
    --python /opt/venv/bin/python \
    --no-deps "${wheel_directory}"/*.whl
uv pip check --python /opt/venv/bin/python
/opt/venv/bin/python -c \
    "import importlib.util; assert importlib.util.find_spec('pip') is None"
test ! -e /opt/venv/bin/pip
test -z "$(find /opt/venv -type f -name 'pip-*.whl' -print -quit)"
test -z "$(find /opt/venv \( -type f -o -type d \) -perm /022 -print -quit)"
