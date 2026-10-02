#!/bin/sh
set -eu

if [ "$#" -ne 2 ]; then
    echo "usage: $0 REQUIREMENTS PYTORCH_INDEX" >&2
    exit 2
fi

requirements=$1
pytorch_index=$2
site_packages=/opt/venv/lib/python3.12/site-packages

test -s "${requirements}"
uv pip install --python /opt/venv/bin/python --require-hashes \
    --index https://pypi.org/simple \
    --default-index "${pytorch_index}" \
    --index-strategy unsafe-first-match \
    --requirement "${requirements}"
uv pip check --python /opt/venv/bin/python
find /opt/venv -type f -name 'pip-*.whl' -delete

# Remove components the batch runtime cannot use and that carry findings with
# no available upgrade. Ray's bundled jar only serves Java workers (Ray 2.59.0
# still bundles Jackson 2.18.8). Its vendored aiohttp is a fallback for when no
# aiohttp is installed; the locked aiohttp is newer. virtualenv is declared by
# ray[serve] but used only by the virtualenv runtime_env plugin, which needs
# pip, and this image ships no pip; conda-lock caps it below the fixed 21.x.
# Local Ray tasks, actors, runtime_env env_vars, and Ray Serve still work.
rm -rf \
    "${site_packages}/ray/jars" \
    "${site_packages}/ray/_private/runtime_env/agent/thirdparty_files" \
    "${site_packages}/virtualenv" \
    "${site_packages}"/virtualenv-*.dist-info
test ! -e "${site_packages}/ray/jars"
test ! -e "${site_packages}/ray/_private/runtime_env/agent/thirdparty_files"
test -z "$(find "${site_packages}" -maxdepth 1 -name 'virtualenv*' -print -quit)"
/opt/venv/bin/python -c "import ray, ray.serve"

chmod -R go-w /opt/venv
