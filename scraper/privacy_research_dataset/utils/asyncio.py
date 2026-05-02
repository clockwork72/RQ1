from __future__ import annotations

import asyncio
from typing import Awaitable, TypeVar

T = TypeVar("T")


async def wait_for_with_cancel_grace(
    awaitable: Awaitable[T],
    *,
    timeout_s: float,
    cancel_grace_s: float,
) -> T:
    task = asyncio.create_task(awaitable)
    try:
        return await asyncio.wait_for(asyncio.shield(task), timeout=timeout_s)
    except asyncio.TimeoutError:
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=cancel_grace_s)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass
        except Exception:
            pass
        raise
    except asyncio.CancelledError:
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=cancel_grace_s)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass
        except Exception:
            pass
        raise
    except Exception:
        if task.done():
            try:
                task.result()
            except Exception:
                pass
        raise
