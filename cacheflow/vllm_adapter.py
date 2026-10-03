"""vLLM V1 custom Scheduler adapter.

Launch with::

    --scheduler-cls cacheflow.vllm_adapter.CARASScheduler

The vLLM scheduler interface is explicitly non-public.  This module performs
fail-fast capability checks and keeps all correctness-critical scheduling in
the upstream ``Scheduler`` implementation.
"""

from __future__ import annotations

import inspect
import time
from typing import Any

from cacheflow.config import CARASConfig
from cacheflow.features import VLLMFeatureProvider
from cacheflow.metrics import PolicyMetrics
from cacheflow.request_queue import CARASRequestQueue

try:
    from vllm.v1.core.sched.scheduler import Scheduler as _VLLMScheduler
except ImportError as exc:  # pragma: no cover - exercised only without vLLM
    _IMPORT_ERROR: ImportError | None = exc

    class _VLLMScheduler:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError(
                "CARASScheduler requires vLLM on Linux. Install a supported "
                "vLLM release before selecting this scheduler."
            ) from _IMPORT_ERROR

else:
    _IMPORT_ERROR = None


class CARASScheduler(_VLLMScheduler):
    """Change only waiting-request admission order; delegate everything else."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._check_capabilities()
        self.cacheflow_config = CARASConfig.from_env()
        provider = VLLMFeatureProvider(
            kv_cache_manager=self.kv_cache_manager,
            block_size=int(self.block_size),
            config=self.cacheflow_config,
            clock=time.time,
        )
        policy_value = getattr(self.policy, "value", str(self.policy)).lower()
        self.waiting = CARASRequestQueue(
            policy=policy_value,
            config=self.cacheflow_config,
            feature_provider=provider,
        )
        self.cacheflow_metrics = PolicyMetrics(
            self.cacheflow_config.metrics_path,
            self.cacheflow_config.metrics_interval,
        )

    def schedule(self, *args: Any, **kwargs: Any) -> Any:
        result = self.waiting.prepare()
        self.cacheflow_metrics.record(result)
        return super().schedule(*args, **kwargs)

    def _check_capabilities(self) -> None:
        missing = [
            name
            for name in ("waiting", "policy", "kv_cache_manager", "block_size")
            if not hasattr(self, name)
        ]
        manager = getattr(self, "kv_cache_manager", None)
        if manager is not None:
            if not hasattr(manager, "coordinator"):
                missing.append("kv_cache_manager.coordinator")
            if not hasattr(manager, "prefix_cache_lookup_enabled") and not hasattr(
                manager, "enable_caching"
            ):
                missing.append(
                    "kv_cache_manager.prefix_cache_lookup_enabled/enable_caching"
                )
        coordinator = getattr(manager, "coordinator", None)
        if coordinator is not None and not hasattr(
            coordinator, "find_longest_cache_hit"
        ):
            missing.append("kv_cache_manager.coordinator.find_longest_cache_hit")
        if missing:
            raise RuntimeError(
                "Unsupported vLLM scheduler internals; missing: "
                + ", ".join(sorted(missing))
            )
        signature = inspect.signature(super().schedule)
        if not isinstance(signature, inspect.Signature):  # defensive
            raise RuntimeError("Unable to inspect vLLM Scheduler.schedule")
