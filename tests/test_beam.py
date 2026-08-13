"""The beam path takes explicit arguments, not a duck-typed opts object."""

import inspect

import numpy as np
import pytest

from spimple.utils.beam import extract_dde_info, interpolate_beam, make_power_beam


@pytest.mark.parametrize(
    ("func", "expected"),
    [
        (interpolate_beam, {"beam_model", "ms", "field", "sparsify_time", "corr_type", "nthreads"}),
        (extract_dde_info, {"ms", "field", "sparsify_time"}),
        (make_power_beam, {"beam_model", "corr_type", "nthreads"}),
    ],
)
def test_signature_is_explicit(func, expected):
    """No 'opts' parameter survives; the six settings are named keyword-only args."""
    params = inspect.signature(func).parameters
    assert "opts" not in params
    assert expected <= set(params)
    assert all(params[name].kind is inspect.Parameter.KEYWORD_ONLY for name in expected)


def test_missing_field_is_a_typeerror_not_an_attributeerror():
    """The spifit bug: field was never set on the throwaway BeamOpts object."""
    with pytest.raises(TypeError):
        extract_dde_info(np.array([1.0e9]), ms=["fake.ms"], sparsify_time=10, nonexistent_kwarg=0)
