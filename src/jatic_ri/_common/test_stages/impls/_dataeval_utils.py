import torch
import torchvision as tv
from torch import nn


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(self.preprocess(x))


@torch.no_grad
def extract_embeddings(images: torch.Tensor, *, embedding_net: EmbeddingNet, batch_size: int = 64) -> torch.Tensor:
    """Extract embeddings from stacked images

    !!! note

        See [this `dataeval` tutorial](https://dataeval.readthedocs.io/en/latest/tutorials/Data_Monitoring.html)
        for usage example

    Args:
        images: Stacked images `[N, C, H, W]`
        embedding_net: [torch.nn.Module][] that performs the embedding
        batch_size: Number of images to be processed concurrently

    """
    embedding_net.eval()
    return torch.vstack([embedding_net(batch) for batch in images.split(batch_size)])
