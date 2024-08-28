# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0
import warnings

_grouped_gemm_is_available: bool = False
try:
    import grouped_gemm
    _grouped_gemm_is_available = True
except ImportError as error:
    warnings.warn('Grouped GEMM not available.')


def grouped_gemm_is_available():
    return _grouped_gemm_is_available


def assert_grouped_gemm_is_available():
    msg = (
        'Grouped GEMM not available. Please run '
        '`pip install git+https://github.com/tgale96/grouped_gemm@main`.',
    )
    assert _grouped_gemm_is_available, msg


backend = grouped_gemm.backend if grouped_gemm_is_available() else None
ops = grouped_gemm.ops if grouped_gemm_is_available() else None
