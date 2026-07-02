# Releasing checkmaite

Use the manual publish jobs only after the release tag already exists on `main`. Pushing the tag does not publish to TestPyPI or PyPI by itself.

## One-time setup

- Protect release tags, for example `[0-9]*`, with create access limited to Maintainers and delete access set to No one.
- Protect the `pypi-production` environment and require approvals before deployment.

Production PyPI publishing uses PyPI Trusted Publishing / GitLab OIDC; do not configure a long-lived `PYPI_API_TOKEN` CI variable for production uploads.

TestPyPI publishing uses a GitLab CI/CD variable named `TEST_PYPI_API_TOKEN` because PyPI/TestPyPI only trust `gitlab.com` as a GitLab OIDC issuer and do not trust self-managed GitLab instances such as `gitlab.jatic.net`. Store the token as a masked/protected variable.

## Release steps

1. Create and push an unprefixed release-shaped version tag, for example `1.2.3`, from a commit on `main`.
2. In GitLab, open **CI/CD → Pipelines → Run pipeline**.
3. Select branch `main`, set `RELEASE_TAG` to the tag, and create the web pipeline.
4. In the pipeline, trigger the manual `publish-testpypi` job. This publishes to TestPyPI and saves the built `dist/` artifacts.
5. Pull the release from TestPyPI and confirm the package works properly. The production `publish-pypi` job remains manual and blocked until `publish-testpypi` succeeds.
6. Go back to the CI pipeline and trigger the manual `publish-pypi` job. Approve the protected `pypi-production` deployment if prompted.
7. Verify the final package at <https://pypi.org/project/checkmaite/>.
8. Pull the release from PyPI and confirm the package works properly.
