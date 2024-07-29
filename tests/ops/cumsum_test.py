# Copyright 2024 MosaicML MegaBlocks authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from megablocks import ops

CUMSUM_TESTS = (
    (1, 32),
    (2, 32),
    (2, 1024),
    (4, 1024),
    (8, 1024),
    (16, 1024),
    (32, 1024),
    (64, 1024),
    (128, 1024),
    (2, 16384),
    (4, 16384),
    (8, 16384),
    (16, 16384),
    (32, 16384),
    (64, 16384),
    (128, 16384),
)


@pytest.mark.gpu
@pytest.mark.parametrize(('n', 'm'), CUMSUM_TESTS)
def test_exclusive_cumsum(n: int, m: int):
    x = torch.randint(0, 2, (n, m)).long().cuda()
    out = ops.exclusive_cumsum(x, 1) * x
    expected_out = (torch.cumsum(x, dim=1) - 1) * x
    assert torch.all(torch.eq(out, expected_out))


@pytest.mark.gpu
@pytest.mark.parametrize(('n', 'm'), CUMSUM_TESTS)
def test_inclusive_cumsum(n: int, m: int):
    x = torch.randint(0, 2, (n, m)).long().cuda()
    out = ops.inclusive_cumsum(x, 1)
    expected_out = torch.cumsum(x, dim=1)
    assert torch.all(torch.eq(out, expected_out))
