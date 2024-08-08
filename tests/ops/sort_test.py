# Copyright 2024 Databricks MegaBlocks authors
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Optional, Union

import numpy as np
import pytest
import torch

from megablocks import ops

SORT_TESTS = [
    (32, torch.int16, None),
    (1024, torch.int16, None),
    (16384, torch.int16, None),
    (32, torch.int32, None),
    (1024, torch.int32, None),
    (16384, torch.int32, None),
    (32, torch.int64, None),
    (1024, torch.int64, None),
    (16384, torch.int64, None),
    (32, torch.int16, 128),
    (1024, torch.int16, 128),
    (16384, torch.int16, 128),
    (32, torch.int32, 128),
    (1024, torch.int32, 128),
    (16384, torch.int32, 128),
    (32, torch.int64, 128),
    (1024, torch.int64, 128),
    (16384, torch.int64, 128),
]


def torch_to_numpy_dtype(dtype: torch.dtype,) -> Union[np.int16, np.int32, np.int64]:
    types: Dict[torch.dtype, Union[np.int16, np.int32, np.int64]] = {
        torch.int16: np.int16,
        torch.int32: np.int32,
        torch.int64: np.int64,
    }
    return types[dtype]


@pytest.mark.gpu
@pytest.mark.parametrize(
    ('n', 'dtype', 'max_val'),
    SORT_TESTS,
)
def test_sort(n: int, dtype: torch.dtype, max_val: Optional[int]):
    if max_val is None:
        max_val = np.iinfo(torch_to_numpy_dtype(dtype)).max
    end_bit = int(np.ceil(np.log2(max_val)))
    x = torch.randint(0, max_val, (n,)).cuda().to(dtype)

    out, indices = ops.sort(x, end_bit)
    expected_out, expected_indices = torch.sort(x)
    assert torch.all(torch.eq(out, expected_out))

    # NOTE: The indices can be in different order depending
    # on sort stability if multiple values in the array are
    # equal.
    data = torch.empty_like(x)
    data.scatter_(0, indices.long(), out)
    expected_data = torch.empty_like(x)
    expected_data.scatter_(0, expected_indices, expected_out)
    assert torch.all(torch.eq(data, expected_data))
