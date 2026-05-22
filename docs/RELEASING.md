# Releasing checkmaite

Use the manual publish jobs only after the release tag already exists on `main`.

## One-time GitLab setup

- Protect release tags, for example `[0-9]*`, with create access limited to Maintainers.
- Add `TEST_PYPI_API_TOKEN` scoped to the `pypi-test` environment.
- Add `PYPI_API_TOKEN` as Protected + Masked and scoped to the `pypi-production` environment.
- Protect the `pypi-production` environment and require approvals before deployment.

## Release steps

1. Create and push an unprefixed version tag, for example `1.2.3`, from a commit on `main`.
2. In GitLab, open **CI/CD → Pipelines → Run pipeline**.
3. Select branch `main`, set `RELEASE_TAG` to the tag, and create the web pipeline.
4. Run `publish-testpypi` first and inspect the saved `dist/` artifacts.
5. After the TestPyPI package looks correct, run and approve `publish-pypi`.
6. Verify the final package at <https://pypi.org/project/checkmaite/>.
