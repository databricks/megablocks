# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0

import torch


def round_up(x: torch.Tensor, value: int):
    assert isinstance(value, int)
    assert x.dtype == torch.int32

    # TODO(tgale): If this becomes and issue
    # do this in a custom kernel. We only expect
    # to use this on arrays of less than 1k elements.
    return torch.div(x + (value - 1), value, rounding_mode='trunc') * value
