# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Shard Descriptor template.

It is recommended to perform tensor manipulations using numpy.
"""
from typing import List

import numpy as np
from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor, ShardDataset


class LocalShardDataset(ShardDataset):
    def __init__(self, data):
        self.features = data[:, :-1]
        self.labels = data[:, -1]
        self.sample_shape = data.shape[-1] - 1

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int):
        return self.features[index], self.labels[index]


class LocalShardDescriptor(ShardDescriptor):
    """Shard descriptor subclass."""

    def get_dataset(self, dataset_type: str) -> ShardDataset:
        return self.dataset

    @property
    def sample_shape(self) -> List[str]:
        return [str(self.dataset.sample_shape)]

    @property
    def target_shape(self) -> List[str]:
        return [str(1)]

    def __init__(self, data_path: str, sample_shape: tuple, target_shape: tuple) -> None:
        """
        Initialize local Shard Descriptor.

        Parameters are arbitrary, set up the ShardDescriptor-related part
        of the envoy_config.yaml as you need.
        """
        super().__init__()
        data = np.loadtxt(data_path, delimiter=',', dtype=np.float)
        self.dataset = LocalShardDataset(data)
