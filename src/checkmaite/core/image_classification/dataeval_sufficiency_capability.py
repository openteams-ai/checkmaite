import logging
from collections.abc import Sequence

import maite.protocols.image_classification as ic
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as tv
from dataeval import config
from dataeval.protocols import Dataset as DatasetType
from dataeval.protocols import TrainingStrategy
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Subset
from torchvision import models

from checkmaite.core._common.dataeval_sufficiency_capability import (
    DataevalSufficiencyBase,
    DataevalSufficiencyConfig,
    SufficiencyDatum,
    SufficiencyTransform,
    _as_torch_dataset,
)

# This is needed to be compatible with min pytorch version 2.2.0
try:
    # This should work for pytorch v2.3 and later
    from torch.amp import GradScaler  # pyright: ignore[reportPrivateImportUsage]

    has_cpu_grad_scaler = True
except ImportError:
    from torch.cuda.amp import GradScaler

    has_cpu_grad_scaler = False


# Workaround for CPU grad scaler unavailable in pytorch 2.2.0
class MockGradScaler:
    def __init__(self, enabled: bool) -> None:
        pass

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss

    def step(self, optimizer: Optimizer) -> None:
        optimizer.step()

    def update(self) -> None:
        pass


logger = logging.getLogger(__name__)


def _maybe_to_tensor(datapoint: SufficiencyDatum) -> SufficiencyDatum:
    img, target, datum_metadata = datapoint
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)
    return img, target, datum_metadata


class _DefaultTrainingStrategy:
    # We can't use python dataclass due to dataeval config serialization internal implementation
    # dataclass becomes a dict internally and train method is discarded
    def __init__(
        self,
        batch_size: int,
        device: torch.device,
        num_epochs: int | None = None,
        num_iters: int | None = None,
        num_workers: int = 4,
        learning_rate: float = 0.03,
        criterion: nn.Module | None = None,
        use_amp: bool = True,
        verbose: bool = True,
    ) -> None:
        self.batch_size = batch_size
        self.device = device
        self.num_epochs = num_epochs
        self.num_iters = num_iters
        self.num_workers = num_workers
        self.learning_rate = learning_rate
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        self.use_amp = use_amp
        self.verbose = verbose

        self.trial_index = 0

    def train(self, model: nn.Module, dataset: DatasetType[SufficiencyDatum], indices: Sequence[int]) -> None:
        self.trial_index += 1
        config.set_seed(self.trial_index, all_generators=True)

        optimizer = optim.SGD(
            model.parameters(),
            lr=self.learning_rate,
            momentum=0.9,
            nesterov=True,
            weight_decay=1e-4,
        )

        use_cuda = "cuda" in self.device.type

        # Define the dataloader for training
        dataloader = DataLoader(
            Subset(_as_torch_dataset(dataset), indices),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=use_cuda,
            drop_last=True,
            shuffle=True,
        )

        if self.num_iters is not None:
            num_iters = self.num_iters
        elif self.num_epochs is not None:
            num_iters = self.num_epochs * len(dataloader)
        else:
            raise ValueError(
                "Must specify either 'num_epochs' or 'num_iters'. "
                "One of these parameters is required for the training strategy."
            )

        # Train for a fixed number of epochs (original behavior)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            total_steps=num_iters,
        )

        autocast = torch.autocast
        # This is needed to be compatible with pytorch 2.2.0
        if has_cpu_grad_scaler:
            scaler = GradScaler(device=self.device.type, enabled=self.use_amp)  # pyright: ignore[reportCallIssue]
        elif use_cuda:
            # old pytorch version with cuda.amp.GradScaler only
            scaler = GradScaler(enabled=self.use_amp)
        else:
            self.use_amp = False
            scaler = MockGradScaler(enabled=self.use_amp)

        model.train()

        cur_iter = 0
        dataloader_iter = iter(dataloader)

        while cur_iter < num_iters:
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                # Reset dataloader if exhausted
                dataloader_iter = iter(dataloader)
                batch = next(dataloader_iter)
                if self.verbose:
                    logger.info(f"{self.trial_index}| iter: {cur_iter} / {num_iters} - {len(dataloader)=}")

            x = batch[0].to(device=self.device, non_blocking=True)
            y = torch.argmax(batch[1].to(dtype=torch.int, device=self.device, non_blocking=True), dim=1)
            with autocast(self.device.type, enabled=self.use_amp):
                outputs = model(x)
                loss = self.criterion(outputs, y)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()

            cur_iter += 1


class DataevalSufficiency(DataevalSufficiencyBase[ic.Dataset, ic.Model]):
    """Estimates dataset sufficiency for a single dataset and a metric. This capability
    gives an estimate of the dataset size required at most to reach a target metric value using
    a baseline model (ResNet18 is used for image classification).
    Metric values are computed on an evaluation subset of the given dataset.

    Note: It is possible that the target metric can be reached by more powerful models
    on smaller datasets, i.e. the size less than the estimated one by this capability.

    Note: execution time of this capability can be large as internally a small neural network
    is trained on data subsets.

    Example of MarkDown report running the capability on CIFAR10 dataset:

    ```
    # Sufficiency Analysis Report

    ## Table of Contents

    - [Sufficiency Analysis](#sufficiency-analysis)


    ---

    ## Dataset Sufficiency Analysis

    **Description:** Dataset Sufficiency estimates the size of the training
    dataset required to achieve target metric value on an evaluation set.


    ### Summary

    | Description | Value |
    | --- | --- |
    | Target dataset size | 234694 |
    | Target metric name | accuracy |
    | Target metric value | 0.92 |

    ### Sufficiency estimation details

    | step | accuracy |
    | --- | --- |
    | 3750.0 | 0.5294399857521057 |
    | 7500.0 | 0.6135200262069702 |
    | 11250.0 | 0.6607999801635742 |
    | 15000.0 | 0.6888800263404846 |
    | 18750.0 | 0.7130399942398071 |
    | 22500.0 | 0.7263200283050537 |
    | 26250.0 | 0.7389600276947021 |
    | 30000.0 | 0.7530400156974792 |
    | 33750.0 | 0.7572000026702881 |
    | 37500.0 | 0.7626399993896484 |

    ![Sufficiency Visualization](/tmp/tmpumqesyqv.png)
    ```

    """

    def _get_model_and_preprocess_fns(
        self,
        num_classes: int,
        image_size: int,
    ) -> tuple[nn.Module, SufficiencyTransform, SufficiencyTransform]:
        # we assume RGB input images
        num_channels = 3
        image_size = max(min(int(0.85 * image_size), 256), 32)
        model = models.resnet18(num_classes=num_classes)
        model.conv1 = nn.Conv2d(num_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        train_preprocess = tv.Compose(
            [
                _maybe_to_tensor,
                tv.Resize(int(image_size * 1.15)),
                tv.RandomCrop(image_size, fill=128),
                tv.RandomHorizontalFlip(),
                tv.ToDtype(dtype=torch.float32, scale=True),
                tv.Normalize(mean=(0.5,) * num_channels, std=(0.25,) * num_channels),
            ]
        )
        eval_preprocess = tv.Compose(
            [
                _maybe_to_tensor,
                tv.Resize(image_size),
                tv.CenterCrop(image_size),
                tv.ToDtype(dtype=torch.float32, scale=True),
                tv.Normalize(mean=(0.5,) * num_channels, std=(0.25,) * num_channels),
            ]
        )
        return model, train_preprocess, eval_preprocess

    def _get_training_strategy(self, config: DataevalSufficiencyConfig) -> TrainingStrategy[SufficiencyDatum]:
        return _DefaultTrainingStrategy(
            num_epochs=config.num_epochs,
            num_iters=config.num_iters,
            batch_size=config.batch_size,
            device=config.device,
            verbose=config.verbose,
            use_amp=config.use_amp,
        )
