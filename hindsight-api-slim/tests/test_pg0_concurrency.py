"""Concurrency tests for ``EmbeddedPostgres.ensure_running``.

Two Starlette lifespans (``multi_bank`` + ``single_bank``) both call
``MemoryEngine.initialize()`` -> ``pg0.ensure_running()`` at app startup.
Without serialization, both callers race the start/is_running/start cycle
and the second's pg0 sees the first's still-launching process, failing
with ``Error: Instance already running (pid: <N>)``. The per-name
asyncio.Lock added in ``ensure_running`` makes the second caller wait,
observe pg0 as running, and reuse it.

These tests pin that behaviour without booting a real pg0 — start/
is_running/get_uri are mocked. The contract: across N concurrent
ensure_running() calls for the same instance name, ``start()`` is
invoked at most once.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from hindsight_api.pg0 import EmbeddedPostgres, _ensure_running_locks


@pytest.mark.asyncio
async def test_concurrent_ensure_running_starts_pg0_once_per_name() -> None:
    """Five concurrent callers for the same name -> start() called once.

    Models the chained-lifespan multi_bank+single_bank case (N=2) plus a
    margin for any future caller composition.
    """
    pg = EmbeddedPostgres(name="hindsight-test-concurrent")
    pg.start = AsyncMock(return_value="postgresql://test/db")  # type: ignore[method-assign]
    pg.get_uri = AsyncMock(return_value="postgresql://test/db")  # type: ignore[method-assign]
    # First caller sees not-running, takes the start path; once the lock
    # releases, subsequent callers observe running and skip start.
    is_running_states = iter([False, True, True, True, True])
    pg.is_running = AsyncMock(side_effect=lambda: next(is_running_states))  # type: ignore[method-assign]

    results = await asyncio.gather(*(pg.ensure_running() for _ in range(5)))

    assert all(r == "postgresql://test/db" for r in results)
    assert pg.start.await_count == 1, "start() must be called at most once across concurrent callers"
    # 1 call sees False (-> start path), 4 see True (-> get_uri path).
    assert pg.get_uri.await_count == 4


@pytest.mark.asyncio
async def test_ensure_running_lock_is_per_name() -> None:
    """Different instance names get independent locks.

    Two ``EmbeddedPostgres`` instances with different names should not
    artificially serialize — they manage independent pg0 processes on
    the filesystem and contend on nothing.
    """
    pg_a = EmbeddedPostgres(name="lock-test-alpha")
    pg_b = EmbeddedPostgres(name="lock-test-beta")
    for pg in (pg_a, pg_b):
        pg.is_running = AsyncMock(return_value=True)  # type: ignore[method-assign]
        pg.get_uri = AsyncMock(return_value=f"postgresql://test/{pg.name}")  # type: ignore[method-assign]
        pg.start = AsyncMock(return_value=f"postgresql://test/{pg.name}")  # type: ignore[method-assign]

    await asyncio.gather(pg_a.ensure_running(), pg_b.ensure_running())

    # Each name registered its own Lock instance.
    assert "lock-test-alpha" in _ensure_running_locks
    assert "lock-test-beta" in _ensure_running_locks
    assert _ensure_running_locks["lock-test-alpha"] is not _ensure_running_locks["lock-test-beta"]
