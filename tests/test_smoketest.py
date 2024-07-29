# Copyright 2024 MosaicML MegaBlocks authors
# SPDX-License-Identifier: Apache-2.0

from megablocks.layers import dmoe, moe


# This very simple test is just to use the above imports, which check and make sure we can import all the top-level
# modules from Megablocks. This is mainly useful for checking that we have correctly conditionally imported all optional
# dependencies.
def test_smoketest():
    assert moe
    assert dmoe
