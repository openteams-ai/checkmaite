# JATIC Reference Implementation

The JATIC Reference Implementation package, `jatic_ri` is set of tools to make implementing the suite of JATIC products easier.

## Description

TODO: Insert detailed description

## Installation

To install the JATIC Reference Implementation package, you will need to [set up a Personal Access Token in Gitlab](https://docs.gitlab.com/ee/user/profile/personal_access_tokens.html) with `read_repository`. Once that is complete, use it to install the package directly from Gitlab: 

```bash
pip install git+https://{username}:{PAT}@gitlab.jatic.net/jatic/increment-demos/reference-implementation.git@main
```

where `{username}` is your Gitlab username and `{PAT}` is the PAT you just created (e.g. starts with "glpat-") and `main` (the suffix) is the branch or git tag.

Alternately, if you have a local copy of the repository, you can run:

```bash
pip install -e .
```

You will be prompted for your username and password. Your password will the be PAT you created above. 

For testing purposes, you'll also need to install [git LFS](https://git-lfs.com/) and run these two commands to ensure that larger testing-only files are checked out prior to actual running testing:
```bash
git lfs install 
git lfs pull
```

Finally, if you want to install this package as part of a conda environment yaml specification, you can add a `pip` section to yaml. Note that you will need to tweak the syntax slightly:

```environment.yml
- dependencies:
  - pip
  - pip:
    - jatic_ri @ git+https://{username}:{PAT}@gitlab.jatic.net/jatic/increment-demos/reference-implementation.git@main
```

## Usage

TODO

## Support

TODO

## Contributing

The JATIC Reference Implementation team welcomes contributions of all forms - questions, documentation, and code contributions. Please visit our [contribution guide](https://gitlab.jatic.net/jatic/increment-demos/reference-implementation/blob/main/contributing.md).

## Authors and acknowledgment

This project has been created and maintained by CDAO JATIC.

## License

TODO

## Documentation

The documentation for this project uses `mkdocs`. 

For active development of the documentation, start a live-reloading docs server:

`mkdocs serve`

For production documentation builds, build a static website using 

`mkdocs build -d public`

where `public` is the output folder. The main page in this case will be at `public/index.html`.

NOTE: `mkdocs` does not follow pages that are being dynamically generated from `*ipynb` via the `mkdocs-jupyter` plugin. 
For this reason, building the docs displays several warnings that can be safely ignored. For example:
`WARNING -  Doc file 'how-tos/index.md' contains a link 'dataset_analysis_configuration.html', but the target 'how-tos/dataset_analysis_configuration.html' is not found among documentation files.`
