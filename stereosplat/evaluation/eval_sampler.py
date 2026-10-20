# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Index sharding for distributed evaluation that does not repeat samples."""

from __future__ import annotations

from collections.abc import Iterator, Sized
from typing import cast

from torch.utils.data import Dataset, DistributedSampler


class EvalSampler(DistributedSampler):
    """Deterministic, duplicate-free shards for distributed evaluation.

    ``DistributedSampler`` pads the index list to a multiple of ``world_size``
    by repeating the head of the dataset. Those repeats are double-weighted in
    the metric average, which makes results depend on ``world_size``. Rank r
    instead takes exactly ``range(r, N, world_size)``: the union over ranks is
    ``[0, N)`` with no duplicates, and local lengths differ by at most one.

    Unequal local lengths are safe here only because evaluation runs every
    collective AFTER the batch loop, never inside it.
    """

    def __init__(
        self,
        dataset: Sized,
        num_replicas: int | None = None,
        rank: int | None = None,
    ) -> None:
        # DistributedSampler is annotated to take a Dataset, but its
        # implementation only ever calls len() on the argument, so Sized is the
        # honest input type. Cast to satisfy the narrower stub.
        super().__init__(
            cast(Dataset, dataset),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        self._n: int = len(dataset)

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, self._n, self.num_replicas))

    def __len__(self) -> int:
        return len(range(self.rank, self._n, self.num_replicas))
