"""Fail-fast static check for the installed vLLM scheduler compatibility surface."""

from __future__ import annotations

import argparse
import inspect
from importlib.metadata import version


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    try:
        from vllm.v1.core.kv_cache_manager import KVCacheManager
        from vllm.v1.core.sched.request_queue import RequestQueue
        from vllm.v1.core.sched.scheduler import Scheduler
    except ImportError as exc:
        raise SystemExit(f"vLLM V1 is not importable: {exc}") from exc

    requirements = {
        "Scheduler.schedule": hasattr(Scheduler, "schedule"),
        "KVCacheManager cache eligibility guard": hasattr(
            KVCacheManager, "prefix_cache_lookup_enabled"
        )
        or "enable_caching" in inspect.signature(KVCacheManager.__init__).parameters,
        "RequestQueue.add_request": hasattr(RequestQueue, "add_request"),
        "RequestQueue.prepend_request": hasattr(RequestQueue, "prepend_request"),
        "RequestQueue.remove_requests": hasattr(RequestQueue, "remove_requests"),
    }
    failed = [name for name, present in requirements.items() if not present]
    if failed:
        raise SystemExit("incompatible vLLM; missing " + ", ".join(failed))
    print("vLLM version:", version("vllm"))
    print("vLLM static compatibility surface: OK (GPU behavior not checked)")
    print("Scheduler.schedule", inspect.signature(Scheduler.schedule))


if __name__ == "__main__":
    main()
