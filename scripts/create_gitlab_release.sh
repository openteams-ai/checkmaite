#!/bin/sh
# Create the GitLab Release for a tag already published to PyPI.
# Notes come from the tag's CHANGELOG.md section; assets link to PyPI.
# Does nothing if the Release already exists, so the job can be retried.
set -eu

tag="$1"
project="${CI_PROJECT_PATH:-jatic/orchestration-interoperability/checkmaite}"

if glab release view "$tag" -R "$project" >/dev/null 2>&1; then
  echo "Release $tag already exists; nothing to do."
  exit 0
fi

changes="$(git show "$tag:CHANGELOG.md" | awk -v heading="## [$tag]" '
  index($0, heading) == 1 { found = 1; next }
  found && /^## \[/ { exit }
  found
')"
if [ -z "$(printf '%s' "$changes" | tr -d '[:space:]')" ]; then
  echo "No '## [$tag]' section in CHANGELOG.md at $tag." >&2
  exit 1
fi

previous="$(git describe --tags --abbrev=0 "$tag^")"
notes="## $tag
$changes

### Installation

\`\`\`bash
pip install checkmaite==$tag
\`\`\`

[Full comparison](https://gitlab.jatic.net/$project/-/compare/$previous...$tag)"

wheel_url="$(curl -fsSL "https://pypi.org/pypi/checkmaite/$tag/json" |
  jq -er '.urls[] | select(.packagetype == "bdist_wheel") | .url')"
links="$(jq -cn --arg tag "$tag" --arg wheel "$wheel_url" '[
  {name: "checkmaite-\($tag)-py3-none-any.whl", url: $wheel, link_type: "package"},
  {name: "checkmaite \($tag) on PyPI", url: "https://pypi.org/project/checkmaite/\($tag)/", link_type: "package"}
]')"

glab release create "$tag" -R "$project" --name "$tag" --notes "$notes" --assets-links "$links"
