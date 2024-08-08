# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from megablocks import ops

_HISTOGRAM_TESTS = (
    (1, 32, torch.int16, 128),
    (1, 1024, torch.int16, 128),
    (1, 16384, torch.int16, 128),
    (1, 32, torch.int32, 128),
    (1, 1024, torch.int32, 128),
    (1, 16384, torch.int32, 128),
    (1, 32, torch.int64, 128),
    (1, 1024, torch.int64, 128),
    (1, 16384, torch.int64, 128),
    (1, 32, torch.int16, 1024),
    (1, 1024, torch.int16, 1024),
    (1, 16384, torch.int16, 1024),
    (1, 32, torch.int32, 1024),
    (1, 1024, torch.int32, 1024),
    (1, 16384, torch.int32, 1024),
    (1, 32, torch.int64, 1024),
    (1, 1024, torch.int64, 1024),
    (1, 16384, torch.int64, 1024),
    (2, 32, torch.int16, 128),
    (2, 1024, torch.int16, 128),
    (2, 16384, torch.int16, 128),
    (2, 32, torch.int32, 128),
    (2, 1024, torch.int32, 128),
    (2, 16384, torch.int32, 128),
    (2, 32, torch.int64, 128),
    (2, 1024, torch.int64, 128),
    (2, 16384, torch.int64, 128),
    (2, 32, torch.int16, 1024),
    (2, 1024, torch.int16, 1024),
    (2, 16384, torch.int16, 1024),
    (2, 32, torch.int32, 1024),
    (2, 1024, torch.int32, 1024),
    (2, 16384, torch.int32, 1024),
    (2, 32, torch.int64, 1024),
    (2, 1024, torch.int64, 1024),
    (2, 16384, torch.int64, 1024),
    (8, 32, torch.int16, 128),
    (8, 1024, torch.int16, 128),
    (8, 16384, torch.int16, 128),
    (8, 32, torch.int32, 128),
    (8, 1024, torch.int32, 128),
    (8, 16384, torch.int32, 128),
    (8, 32, torch.int64, 128),
    (8, 1024, torch.int64, 128),
    (8, 16384, torch.int64, 128),
    (8, 32, torch.int16, 1024),
    (8, 1024, torch.int16, 1024),
    (8, 16384, torch.int16, 1024),
    (8, 32, torch.int32, 1024),
    (8, 1024, torch.int32, 1024),
    (8, 16384, torch.int32, 1024),
    (8, 32, torch.int64, 1024),
    (8, 1024, torch.int64, 1024),
    (8, 16384, torch.int64, 1024),
)


# Override the seed_all fixture in autouse.py because
# _histc_cuda does not have a deterministic implementation
@pytest.fixture()
def seed_all():
    torch.use_deterministic_algorithms(False)
    return


@pytest.mark.gpu
@pytest.mark.parametrize(('m', 'n', 'dtype', 'max_val'), _HISTOGRAM_TESTS)
def test_histogram(m: int, n: int, dtype: torch.dtype, max_val: int):
    x = torch.randint(0, max_val, (m, n)).cuda().to(dtype)

    out = ops.histogram(x, max_val)
    expected_out = torch.stack([torch.histc(y, max_val, 0, max_val - 1) for y in torch.split(x, 1)])
    assert torch.all(torch.eq(out, expected_out))
