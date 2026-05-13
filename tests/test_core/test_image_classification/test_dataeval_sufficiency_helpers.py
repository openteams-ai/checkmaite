import numpy as np
import torch

from checkmaite.core._common.dataeval_sufficiency_capability import DataevalSufficiencyConfig
from checkmaite.core.image_classification.dataeval_sufficiency_capability import (
    DataevalSufficiency,
    MockGradScaler,
    _maybe_to_tensor,
)


def test_dataeval_sufficiency_helpers_preserve_training_invariants() -> None:
    image = np.zeros((3, 4, 5), dtype=np.uint8)
    target = np.array([1, 0])
    metadata = {"id": "datum"}

    converted = _maybe_to_tensor((image, target, metadata))
    assert isinstance(converted[0], torch.Tensor)
    assert converted[1:] == (target, metadata)
    tensor_image = torch.zeros((3, 4, 5))
    assert _maybe_to_tensor((tensor_image, "target", "metadata"))[0] is tensor_image

    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    parameter.grad = torch.tensor([2.0])
    scaler = MockGradScaler(enabled=False)

    loss = torch.tensor(3.0)
    assert scaler.scale(loss) is loss
    scaler.step(optimizer)
    scaler.update()
    assert parameter.item() < 1.0

    capability = DataevalSufficiency()
    model, train_preprocess, eval_preprocess = capability._get_model_and_preprocess_fns(num_classes=2, image_size=16)
    strategy = capability._get_training_strategy(DataevalSufficiencyConfig(batch_size=2, num_iters=1, device="cpu"))

    assert isinstance(model, torch.nn.Module)
    assert all(callable(fn) for fn in (train_preprocess, eval_preprocess))
    assert strategy.batch_size == 2
    assert strategy.device.type == "cpu"
