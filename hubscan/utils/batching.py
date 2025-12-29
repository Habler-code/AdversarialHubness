# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Batching utilities for efficient processing."""

from typing import Iterator, TypeVar, List
import numpy as np

T = TypeVar("T")


def batch_iterator(items: List[T], batch_size: int) -> Iterator[List[T]]:
    """Iterate over items in batches."""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def batch_arrays(arrays: List[np.ndarray], batch_size: int) -> Iterator[np.ndarray]:
    """Batch numpy arrays efficiently."""
    n = len(arrays[0]) if arrays else 0
    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        yield tuple(arr[i:end] for arr in arrays)

