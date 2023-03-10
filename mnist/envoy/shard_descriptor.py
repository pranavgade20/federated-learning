# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Shard Descriptor template.

It is recommended to perform tensor manipulations using numpy.
"""
from typing import List

import numpy as np
import torch
from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor, ShardDataset
from openfl.component.envoy.envoy import Envoy
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision


class LocalShardDataset(ShardDataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        return self.dataset[index][0][0], torch.tensor(self.dataset[index][1])


class LocalShardDescriptor(ShardDescriptor):
    """Shard descriptor subclass."""

    def get_dataset(self, dataset_type: str) -> ShardDataset:
        return self.dataset

    @property
    def sample_shape(self) -> List[str]:
        return ['28', '28']

    @property
    def target_shape(self) -> List[str]:
        return ['10']

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize local Shard Descriptor.

        Parameters are arbitrary, set up the ShardDescriptor-related part
        of the envoy_config.yaml as you need.
        """
        super().__init__()
        transform_mnist = transforms.Compose([transforms.ToTensor()])
        trainset_mnist = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
        # testset_mnist = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)

        self.dataset = LocalShardDataset(trainset_mnist)


