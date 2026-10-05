# System requirements

This page covers the `checkmaite` Python library. For the batch container, see
[Batch container](container.md).

## Software

`checkmaite` requires Python 3.10, 3.11, or 3.12 and PyTorch 2.2.0 or later. The
installer pulls in PyTorch and every other dependency. See
[Setup](../get-started/install_setup.md).

## Hardware

| Resource | Minimum    | Recommended                         |
| -------- | ---------- | ----------------------------------- |
| CPU      | 2 cores    | 4 or more cores                     |
| Memory   | 8 GB       | 16 GB or more                       |
| Disk     | 10 GB free | 25 GB or more free                  |
| GPU      | None       | NVIDIA GPU with 4 GB or more memory |

- **Memory** grows with the dataset. DataEval capabilities keep per-image
  results for the whole dataset in memory, so size memory to your largest
  dataset. Each tutorial in this documentation peaks below 3 GB, and the
  end-to-end [object detection workflow](../get-started/checkmaite_api_od.ipynb)
  completes on 2 CPU cores with 8 GB.
- **Disk** holds the Python environment, downloaded model weights, and your
  data, caches, and results. On Linux the environment is about 6 GB, because
  PyPI's PyTorch includes the CUDA libraries; on macOS it is about 2 GB. The
  models used in the tutorials add about 250 MB.
- **A GPU is optional.** Every capability runs on the CPU. When no device is
  given, `checkmaite` uses CUDA if it is available, then Apple's MPS, then the
  CPU. The CUDA versions and GPUs supported depend on the PyTorch build you
  install.

## Supported architectures

Verified on Linux x86_64 and aarch64, and on macOS arm64 (Apple silicon). macOS
arm64 requires `brew install libomp`; see
[Setup](../get-started/install_setup.md). Other platforms are not tested.

## Internet access

- **Installation** needs PyPI (or a mirror), or conda-forge for conda installs.
  `checkmaite-plugins` installs from `gitlab.jatic.net`.
- **First use of a pretrained model** downloads its weights once: torchvision
  weights from `download.pytorch.org` into `TORCH_HOME` (default
  `~/.cache/torch`), and the VisDrone model from `data.kitware.com` into
  `~/.cache/modelmaite`. To work offline, copy these caches from a connected
  machine.
- **Optional features** use the network only when you configure them: datasets
  or an analytics store on S3 or Google Cloud Storage, and Ray clusters or Ray
  Serve endpoints given by address.

Everything else runs offline. A local Ray runtime also sends anonymous usage
statistics unless you set `RAY_USAGE_STATS_ENABLED=0`.
