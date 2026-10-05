# Security Policy

This document describes how to report vulnerabilities in `CheckMAITE` and how the
maintainers handle findings from automated security scanners.

## Reporting a vulnerability

Please do **not** report security vulnerabilities in an ordinary or public issue. Report them
privately through either channel:

- **JATIC users:** open a
  [new issue](https://gitlab.jatic.net/jatic/orchestration-interoperability/checkmaite/-/issues/new)
  on the JATIC GitLab and select **This issue is confidential** before creating it, so only
  project members can see it.
- **Everyone else:** use GitHub's
  [private vulnerability reporting](https://github.com/openteams-ai/checkmaite/security/advisories/new),
  which is visible only to the maintainers.

Include:

- A description of the issue and its impact
- Steps to reproduce (a minimal script, dataset, or command line is ideal)
- Affected versions or commits
- Your name and (optionally) a credit preference

For non-security bugs, follow the regular process in
[CONTRIBUTING.md](https://gitlab.jatic.net/jatic/orchestration-interoperability/checkmaite/-/blob/main/CONTRIBUTING.md).

## Supported versions

Only the latest minor release receives security fixes.

| Version          | Status      | Security fixes |
| ---------------- | ----------- | -------------- |
| `0.4.x`          | active      | yes            |
| anything older   | unsupported | no             |

## Automated scanner coverage

The CI pipeline runs these security checks (see [.gitlab-ci.yml](https://gitlab.jatic.net/jatic/orchestration-interoperability/checkmaite/-/blob/main/.gitlab-ci.yml)
and [.pre-commit-config.yaml](https://gitlab.jatic.net/jatic/orchestration-interoperability/checkmaite/-/blob/main/.pre-commit-config.yaml)):

| Scanner                                    | Scope                      | Suppression mechanism                                   |
| ------------------------------------------ | -------------------------- | ------------------------------------------------------- |
| Bandit (pre-commit)                        | Python source              | per-line `# nosec <test-id>` with a justification       |
| SAST (JATIC `dr-compliance` component)     | source code, excluding `tests` | GitLab Vulnerability Report dismissal (with comment) |
| Gemnasium dependency scanning              | `uv.lock`                  | GitLab Vulnerability Report dismissal (with comment)    |
| GitLab Secret Detection                    | working tree               | do not commit secrets                                   |

Findings flow into the GitLab Vulnerability Report.

## Dismissing a false positive

The same workflow applies whichever scanner raised the finding.

1. **Verify it is a false positive.** Read the rule or CVE description, inspect the code or
   dependency, and confirm the finding does not apply to how `CheckMAITE` uses the affected
   component. If you are unsure, treat it as a true positive and fix it.

2. **Get a second opinion.** Security suppressions land through a normal MR reviewed by a
   maintainer. The MR description must link to the finding (rule ID or CVE) and to any upstream
   advisory or discussion that supports the dismissal.

3. **Record the suppression in the right place.** The justification must be specific to this
   project. Generic statements like "not exploitable in our case" will be rejected in review.
   - **Bandit:** add `# nosec <test-id>` on the triggering line, with the reason in the same
     comment.
   - **Dependency scanning:** prefer bumping the affected package, or constraining it in
     `[tool.uv].constraint-dependencies` in `pyproject.toml`. When no fix is available upstream,
     dismiss the finding in the GitLab Vulnerability Report with a comment containing:
     1. The CVE ID
     2. Why `CheckMAITE` is not exploitable (for example, the affected API is never called)
     3. A re-evaluation date
   - **SAST:** dismiss the finding in the GitLab Vulnerability Report with the same three-part
     comment.
   - **Secret Detection:** a true positive must be rotated immediately, because the secret is
     already in git history and must be assumed compromised.

4. **Set an expiry.** Every dismissal carries a re-evaluation date (90 days for HIGH or
   CRITICAL, 180 days for MEDIUM). When the date passes, the dismissal must be removed or
   re-justified.

## Hardcoded-secret policy

The project ships no hardcoded secrets. If you discover what looks like a credential,
password, token, or private key in this repository, even in a test fixture, report it through
the confidential channel above rather than an ordinary issue.
