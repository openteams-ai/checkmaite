# Contributing

Thank you for your interest in `CheckMAITE`! The team welcomes contributions of all
forms: bug reports, feature requests, questions, documentation, and code.

## Development Setup

Clone the repository from the JATIC GitLab and create an environment with `uv` or
conda. The [Development Setup Guide](https://openteams-ai.github.io/checkmaite/development/dev_setup.html)
covers both, along with running the tests and pre-commit hooks.

## How Can I Contribute?

### Reporting Bugs

Bugs are tracked as issues in either of two places, and the CheckMAITE team triages both:

- **JATIC users** with a JATIC GitLab account: the
  [GitLab issue board](https://gitlab.jatic.net/jatic/orchestration-interoperability/checkmaite/-/issues).
- **Everyone else:** the public
  [GitHub issue tracker](https://github.com/openteams-ai/checkmaite/issues).

**Do not report security vulnerabilities as ordinary issues.** Follow
[SECURITY.md](https://gitlab.jatic.net/jatic/orchestration-interoperability/checkmaite/-/blob/main/SECURITY.md)
instead.

#### Submitting a Bug Report

Open a new issue with the bug report template, on
[GitLab](https://gitlab.jatic.net/jatic/orchestration-interoperability/checkmaite/-/issues/new?issuable_template=bug-report)
or [GitHub](https://github.com/openteams-ai/checkmaite/issues/new?template=bug_report.yml),
and fill in each section. The template asks for:

```text
Steps to Reproduce:
 1.
 2.
 3.
 ...

Expected Behavior:

Actual Behavior:

Frequency of Behavior:

Environment (CheckMAITE version, Python version, OS, CPU or GPU):
```

#### Writing a Useful Bug Report

Bugs can be difficult to pin down, but the tips below help the maintainers find and
fix them quickly.

- Use a clear and descriptive title
- Describe the exact steps (before and during) which led to the issue
- Provide specific examples, such as the dataset, model, and capability used
- Describe the behavior observed after following each step
- Explain the expected behavior compared to what was observed
- Include full call stacks and error messages when possible

### Feature Requests

The CheckMAITE team encourages other JATIC teams to submit `CheckMAITE` feature requests by
opening an issue with the feature request template, on
[GitLab](https://gitlab.jatic.net/jatic/orchestration-interoperability/checkmaite/-/issues/new?issuable_template=feature-request)
or [GitHub](https://github.com/openteams-ai/checkmaite/issues/new?template=feature_request.yml).

The CheckMAITE team periodically reviews these requests and then divides them according to
scope and applicability across `CheckMAITE`:

- Features which enhance only a specific use-case, or require primarily niche familiarity of a
  JATIC component, are marked to be ***led by the appropriate JATIC team*** (e.g. the
  requestor's team). The CheckMAITE team can coordinate and support these cases. This marking
  is accomplished by adding the appropriate JATIC team's tag to the issue.
- Features determined to be generally applicable to `CheckMAITE` are prioritized for
  development ***led by the CheckMAITE team itself.*** Other JATIC teams may volunteer to
  support or even lead these efforts. These issues are tracked within the CheckMAITE team's
  usual task workflow.

### How Issues Are Managed

Every bug report, feature request, and the team's response to it is kept on the issue itself,
on the [GitLab issue board](https://gitlab.jatic.net/jatic/orchestration-interoperability/checkmaite/-/issues) or the
[GitHub issue tracker](https://github.com/openteams-ai/checkmaite/issues), wherever it was filed.
The CheckMAITE team triages new issues on both. On GitLab, it gives each one a `type::` label and a
`status::` label (for example `status::ready to start`), and links the merge request that resolves
an issue so the issue closes when that merge request is merged. On GitHub, it labels each issue
(for example `bug` or `enhancement`) and responds on the issue.

### Code-Compatibility Refactors

The environment for `CheckMAITE` includes all of the JATIC tools as dependencies. Changes in
`CheckMAITE` code may be required to support changes in these dependencies. The CheckMAITE team
is ultimately responsible for ensuring `CheckMAITE` correctly builds and operates with the set
of dependency versions dictated by technical and security requirements.

The CheckMAITE team requires the collaboration of JATIC teams which release components on which
`CheckMAITE` depends:

- ***A bug or security vulnerability in a JATIC component.*** In this case, the CheckMAITE team
  will communicate with the JATIC Team responsible for the affected component in order to
  coordinate a solution and timeline for the fix.
- ***A new version of a JATIC component introduces breaking changes.*** The CheckMAITE team will
  communicate with the JATIC Team responsible for the component to understand the scope of the
  fix required. The CheckMAITE team can support minor compatibility changes, but will request
  assistance if substantial changes are required to support the new version. ***The CheckMAITE
  team requests maximum advanced notice when breaking API changes will be introduced in
  upcoming versions of JATIC components*** in order to reduce their impact.

### Changelog

`CheckMAITE` tracks all notable changes in
[CHANGELOG.md](https://gitlab.jatic.net/jatic/orchestration-interoperability/checkmaite/-/blob/main/CHANGELOG.md)
following [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/). When opening a merge
request, add one or more lines under the `## [Unreleased]` section at the top of `CHANGELOG.md`.
Use the standard subsections (`Added`, `Changed`, `Deprecated`, `Removed`, `Fixed`,
`Security`) and write each entry from the perspective of an external consumer ("Added X", not
"I added X"). Update the entry during the MR, not after merge, so the changelog is always in
sync with the code.

### External Contribution Procedure

The CheckMAITE team requests that Contributing Teams follow this procedure:

1. Contributing Team commits to the `CheckMAITE` codebase within a feature branch of the
   CheckMAITE repo or their own fork of the repo.
2. Contributing Team opens an MR to the CheckMAITE repo and adds the
   ***"needs:: contributing team review"*** tag.
3. A second member of the Contributing Team completes a review of the MR.
4. A member of the Contributing Team removes the "needs:: contributing team review" tag.
5. A member of the Contributing Team adds the ***"needs: CheckMAITE team review"*** tag.

When the "needs: CheckMAITE team review" tag is added to an MR, the CheckMAITE team will
schedule the MR review and communicate with the owner of the MR on any further changes.
