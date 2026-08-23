from __future__ import annotations

import pytest

from multicam_bench.bench.lag import Anchor, LoopUnwrapper, ingest_lag_s


def test_ingest_lag_on_pace() -> None:
    anchor = Anchor(frame_index=100, t_wall=10.0)
    lag = ingest_lag_s(anchor, frame_index=130, t_recv=11.0, fps_source=30.0)
    assert lag == pytest.approx(0.0)


def test_ingest_lag_behind() -> None:
    anchor = Anchor(frame_index=100, t_wall=10.0)
    lag = ingest_lag_s(anchor, frame_index=130, t_recv=11.05, fps_source=30.0)
    assert lag == pytest.approx(0.05)


def test_ingest_lag_ahead_is_negative() -> None:
    anchor = Anchor(frame_index=0, t_wall=0.0)
    lag = ingest_lag_s(anchor, frame_index=30, t_recv=0.9, fps_source=30.0)
    assert lag == pytest.approx(-0.1)


def test_ingest_lag_uses_embedded_index_not_received_count() -> None:
    # Frames may have been dropped between anchor and this sample, so the received
    # count would under-report elapsed time. Elapsed time must come from the index
    # delta, not from counting how many frames arrived.
    anchor = Anchor(frame_index=0, t_wall=0.0)
    lag = ingest_lag_s(anchor, frame_index=60, t_recv=2.0, fps_source=30.0)
    assert lag == pytest.approx(0.0)


def test_unwrap_single_loop_no_wrap() -> None:
    unwrapper = LoopUnwrapper(period=100)
    result = [unwrapper.unwrap(i) for i in range(50)]
    assert result == list(range(50))


def test_unwrap_across_loop_boundary() -> None:
    unwrapper = LoopUnwrapper(period=100)
    raws = [95, 96, 97, 98, 99, 0, 1, 2]
    result = [unwrapper.unwrap(r) for r in raws]
    assert result == [95, 96, 97, 98, 99, 100, 101, 102]


def test_unwrap_multiple_loops() -> None:
    period = 10
    unwrapper = LoopUnwrapper(period=period)
    raws = list(range(period)) * 3
    result = [unwrapper.unwrap(r) for r in raws]
    assert result == list(range(30))


def test_unwrap_tolerates_dropped_frames_near_boundary() -> None:
    # frames 98, 99, 0, 1 were dropped: the next received raw index is 2
    unwrapper = LoopUnwrapper(period=100)
    unwrapper.unwrap(97)
    assert unwrapper.unwrap(2) == 102


def test_unwrap_ignores_small_backward_noise() -> None:
    # a one-off marker misread should not be mistaken for a loop restart
    unwrapper = LoopUnwrapper(period=100)
    unwrapper.unwrap(50)
    assert unwrapper.unwrap(49) == 49


def test_unwrap_rejects_non_positive_period() -> None:
    with pytest.raises(ValueError, match="period"):
        LoopUnwrapper(period=0)
