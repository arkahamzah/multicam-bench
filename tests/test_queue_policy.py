from __future__ import annotations

import queue

import pytest

from multicam_bench.bench.queue_policy import BoundedDropOldestQueue


def test_put_under_capacity_does_not_drop() -> None:
    q: BoundedDropOldestQueue[int] = BoundedDropOldestQueue(maxsize=1)
    dropped = q.put(1)
    assert dropped is False
    assert q.drops == 0


def test_put_over_capacity_drops_oldest() -> None:
    q: BoundedDropOldestQueue[int] = BoundedDropOldestQueue(maxsize=1)
    q.put(1)
    dropped = q.put(2)
    assert dropped is True
    assert q.drops == 1
    assert q.get(timeout=1.0) == 2  # the newest item survives, not the oldest


def test_drops_are_counted_cumulatively() -> None:
    q: BoundedDropOldestQueue[int] = BoundedDropOldestQueue(maxsize=1)
    for i in range(5):
        q.put(i)
    assert q.drops == 4  # first put fills the queue, the next 4 each evict one


def test_maxsize_larger_than_one_absorbs_without_dropping() -> None:
    q: BoundedDropOldestQueue[int] = BoundedDropOldestQueue(maxsize=3)
    assert q.put(1) is False
    assert q.put(2) is False
    assert q.put(3) is False
    assert q.put(4) is True
    assert q.drops == 1


def test_rejects_non_positive_maxsize() -> None:
    with pytest.raises(ValueError, match="maxsize"):
        BoundedDropOldestQueue(maxsize=0)


def test_get_after_drain_respects_timeout() -> None:
    q: BoundedDropOldestQueue[int] = BoundedDropOldestQueue(maxsize=1)
    with pytest.raises(queue.Empty):
        q.get(timeout=0.05)
