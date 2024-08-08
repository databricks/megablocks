# Copyright 2024 Databricks MegaBlocks authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from tests.conftest import _get_option


@pytest.fixture
def rank_zero_seed(pytestconfig: pytest.Config) -> int:
    """Read the rank_zero_seed from the CLI option."""
    seed = _get_option(pytestconfig, 'seed', default='0')
    return int(seed)
