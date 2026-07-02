# Releasing checkmaite

Use the manual publish jobs only after the release tag already exists on `main`. Pushing the tag does not publish to TestPyPI or PyPI by itself.

## One-time setup

- Protect release tags, for example `[0-9]*`, with create access limited to Maintainers and delete access set to No one.
- Protect the `pypi-production` environment and require approvals before deployment.

Publishing uses GitLab CI/CD variables named `TEST_PYPI_API_TOKEN` and `PYPI_API_TOKEN` because PyPI/TestPyPI have not yet onboarded `gitlab.jatic.net` as a trusted self-managed GitLab issuer. Store both tokens as masked/protected variables.

## Release steps

1. Create and push an unprefixed release-shaped version tag, for example `1.2.3`, from a commit on `main`.
2. In GitLab, open **CI/CD → Pipelines → Run pipeline**.
3. Select branch `main`, set `RELEASE_TAG` to the tag, and create the web pipeline.
4. In the pipeline, trigger the manual `publish-testpypi` job. This publishes to TestPyPI and saves the built `dist/` artifacts.
5. Pull the release from TestPyPI and confirm the package works properly. The production `publish-pypi` job remains manual and blocked until `publish-testpypi` succeeds.
6. Go back to the CI pipeline and trigger the manual `publish-pypi` job. Approve the protected `pypi-production` deployment if prompted.
7. Verify the final package at <https://pypi.org/project/checkmaite/>.
8. Pull the release from PyPI and confirm the package works properly.
