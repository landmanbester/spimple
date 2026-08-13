"""The beam path takes explicit arguments, not a duck-typed opts object."""

import inspect

import numpy as np
import pytest

from spimple.utils.beam import (
    _unflagged_counts,
    extract_dde_info,
    interpolate_beam,
    make_power_beam,
)


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


def test_unflagged_counts_counts_each_timeslot_own_rows():
    """Three timeslots of two rows each, one flagged row in the middle slot."""
    times = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
    flags = np.array([False, False, True, False, False, False])
    _, time_idx, time_counts = np.unique(times, return_index=True, return_counts=True)

    out = _unflagged_counts(flags, time_idx, time_counts, np.zeros(3, dtype=np.int32))

    np.testing.assert_array_equal(out, [2, 1, 2])


def test_unflagged_counts_does_not_read_past_the_end():
    """The old implementation looked up time_idx[i + 1] on the final iteration.

    numba runs this kernel with bounds checking off, so the overrun read
    adjacent memory rather than raising -- the last timeslot's count was
    whatever that garbage produced. Counting to time_idx[i] + time_counts[i]
    removes the lookup entirely.
    """
    times = np.array([1.0, 2.0, 3.0])
    flags = np.zeros(3, dtype=bool)
    _, time_idx, time_counts = np.unique(times, return_index=True, return_counts=True)

    out = _unflagged_counts(flags, time_idx, time_counts, np.zeros(3, dtype=np.int32))

    assert out[-1] == 1, "the final timeslot must be counted from its own rows"


def test_unflagged_counts_ignores_rows_between_subsampled_timeslots():
    """--sparsify-time keeps every Nth timeslot; each still counts only its own rows.

    Slot 0 and slot 2 are kept. Slot 1 sits between them and is fully unflagged;
    if the count spanned to the next kept entry it would leak in and give 4.
    """
    times = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
    flags = np.zeros(6, dtype=bool)
    _, time_idx, time_counts = np.unique(times, return_index=True, return_counts=True)

    keep = np.arange(0, 3, 2)  # slots 0 and 2
    out = _unflagged_counts(flags, time_idx[keep], time_counts[keep], np.zeros(2, dtype=np.int32))

    np.testing.assert_array_equal(out, [2, 2])


def test_unflagged_counts_requires_boolean_flags():
    """Integer flags make `~` a bitwise-not (~0 == -1), producing negative counts."""
    times = np.array([1.0, 1.0])
    _, time_idx, time_counts = np.unique(times, return_index=True, return_counts=True)

    boolean = _unflagged_counts(np.zeros(2, dtype=bool), time_idx, time_counts, np.zeros(1, dtype=np.int32))
    integer = _unflagged_counts(np.zeros(2, dtype=np.int32), time_idx, time_counts, np.zeros(1, dtype=np.int32))

    assert boolean[0] == 2
    assert integer[0] < 0, "guards the astype(bool) at the call site"
