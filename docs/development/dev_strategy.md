# Development Strategies

## Branch strategy

This project uses the GitHub Flow branching strategy. As such, the main branch contains production-ready code. Feature
branches are used to track work on new features or bug fixes and are merged back into the main branch when the work
is complete.

## Merge strategy

The `main` branch is a protected branch. All changes to `main` must go through the Merge Request (MR) process.
Developers should mark their MRs as "Draft" while they are still under development (though they may want to
open an MR early to test on CI). Once the MR is ready and passing all CI, the "Draft" flag can be removed and the
developer should assign reviewers. Approval from a project maintainer is required for merging.

If the original Issue number is included as the prefix of the branch name, GitLab will automatically close the Issue when the MR is closed.

## Release strategy

Releases are named and tagged using semantic versioning, (i.e., `<major>.<minor>.<patch>`). Patch is incremented to indicate bug fixes. Minor is incremented to indicate new functionality that does not change the core API. Major is incremented to indicate compatibility-breaking API changes.


Changes associated with each release should be communicated to the broader JATIC team via changelogs and release notes along with the announcement.
