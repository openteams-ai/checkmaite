import sys
from collections.abc import Iterable

import torch
import torchvision as tv
import torchvision.transforms.v2.functional as tvf
from numpy.typing import ArrayLike
from torch import nn

from jatic_ri._common.models import set_device

if sys.version_info >= (3, 12):
    from itertools import batched
else:
    from collections.abc import Iterator
    from itertools import islice
    from typing import TypeVar

    T = TypeVar("T")

    # See https://docs.python.org/3.12/library/itertools.html#itertools.batched
    def batched(iterable: Iterable[T], n: int) -> Iterator[tuple[T, ...]]:
        # batched('ABCDEFG', 3) → ABC DEF G
        if n < 1:
            raise ValueError("n must be at least one")
        iterator = iter(iterable)
        while batch := tuple(islice(iterator, n)):
            yield batch


class EmbeddingNet(nn.Module):
    """Resnet18 as embedding model

    Turns stacked RGB images `[N, C, H, W]` into `[N, E]`,
    where `E` is `dimensionality`

    !!! note

        See [this `dataeval` tutorial](https://dataeval.readthedocs.io/en/latest/tutorials/Data_Monitoring.html)
        for usage example

    Args:
        dimensionality: Dimensionality of the embeddings

    """

    def __init__(self, dimensionality: int = 128) -> None:
        super().__init__()
        weights = tv.models.ResNet18_Weights.DEFAULT
        self.preprocess = weights.transforms()
        self.model = tv.models.resnet18(weights=weights)
        self.model.fc = nn.Linear(self.model.fc.in_features, dimensionality)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)


@torch.no_grad
def extract_embeddings(
    images: Iterable[ArrayLike],
    *,
    embedding_net: EmbeddingNet,
    batch_size: int = 64,
    device: str | None | torch.device = "cpu",
) -> torch.Tensor:
    """Extract embeddings from images

    !!! note

        See [this `dataeval` tutorial](https://dataeval.readthedocs.io/en/latest/tutorials/Data_Monitoring.html)
        for usage example

    Args:
        images: Iterable of images `[C, H, W]`
        embedding_net: [torch.nn.Module][] that performs the embedding
        batch_size: Number of images to be processed concurrently
        device: Device to transfer the inputs to

    Returns:
        Embeddings of shape `[N, E]`, where `N = len(images)` and `E` is the dimensionality of `embedding_net`

    """
    device = set_device(device)
    embedding_net.to(device).eval()

    def preprocess(image: ArrayLike) -> torch.Tensor:
        return embedding_net.preprocess(
            tvf.to_dtype(torch.as_tensor(image, device=device), dtype=torch.float32, scale=True)
        )

    return torch.vstack(
        [embedding_net(torch.stack(batch)) for batch in batched((preprocess(image) for image in images), batch_size)]
    )
