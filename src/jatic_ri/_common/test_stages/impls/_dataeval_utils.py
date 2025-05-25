import torch
import torchvision.transforms.v2 as v2
from dataeval.typing import Transform
from dataeval.utils.torch.models import ResNet18


def get_resnet18(dim: int | tuple[int, int] = 128) -> tuple[torch.nn.Module, Transform]:
    model = ResNet18().eval()
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((dim, dim) if isinstance(dim, int) else dim),
        ]
    )
    return model, transform
