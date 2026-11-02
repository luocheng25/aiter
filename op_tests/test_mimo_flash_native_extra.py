"""Actual small graph shapes and the largest global-TP8 ACC3 MoE batch."""

import pytest

from op_tests.test_mimo_flash_native_decode import test_flash_native_eager_and_graph as _check


@pytest.mark.parametrize("inter_dim,tokens", [
    (256, 4), (256, 8), (256, 16), (256, 32), (256, 5760),
    (512, 4), (512, 8), (512, 16), (512, 32), (512, 1440),
])
def test_native_graph_and_h200_extremes(inter_dim, tokens):
    _check(inter_dim, tokens)
