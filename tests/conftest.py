# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0

import os
from typing import List, Optional

import pytest
from composer.utils import reproducibility

# Allowed options for pytest.mark.world_size()
WORLD_SIZE_OPTIONS = (1, 2)

# Enforce deterministic mode before any tests start.
reproducibility.configure_deterministic_mode()

# Add the path of any pytest fixture files you want to make global
pytest_plugins = [
    'tests.fixtures.autouse',
    'tests.fixtures.fixtures',
]


def _get_world_size(item: pytest.Item):
    """Returns the world_size of a test, defaults to 1."""
    _default = pytest.mark.world_size(1).mark
    return item.get_closest_marker('world_size', default=_default).args[0]


def _get_option(
    config: pytest.Config,
    name: str,
    default: Optional[str] = None,
) -> str:  # type: ignore
    val = config.getoption(name)
    if val is not None:
        assert isinstance(val, str)
        return val
    val = config.getini(name)
    if val == []:
        val = None
    if val is None:
        if default is None:
            pytest.fail(f'Config option {name} is not specified but is required',)
        val = default
    assert isinstance(val, str)
    return val


def _add_option(
    parser: pytest.Parser,
    name: str,
    help: str,
    choices: Optional[list[str]] = None,
):
    parser.addoption(
        f'--{name}',
        default=None,
        type=str,
        choices=choices,
        help=help,
    )
    parser.addini(
        name=name,
        help=help,
        type='string',
        default=None,
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: List[pytest.Item],
) -> None:
    """Filter tests by world_size (for multi-GPU tests)"""
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    print(f'world_size={world_size}')

    conditions = [
        lambda item: _get_world_size(item) == world_size,
    ]

    # keep items that satisfy all conditions
    remaining = []
    deselected = []
    for item in items:
        if all(condition(item) for condition in conditions):
            remaining.append(item)
        else:
            deselected.append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = remaining


def pytest_addoption(parser: pytest.Parser) -> None:
    _add_option(
        parser,
        'seed',
        help="""\
        Rank zero seed to use. `reproducibility.seed_all(seed + dist.get_global_rank())` will be invoked
        before each test.""",
    )


def pytest_sessionfinish(session: pytest.Session, exitstatus: int):
    if exitstatus == 5:
        session.exitstatus = 0  # Ignore no-test-ran errors
