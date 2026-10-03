from __future__ import annotations

import math
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from cacheflow.config import CARASConfig, SchedulingMode


class ConfigTests(unittest.TestCase):
    def test_empty_environment_uses_defaults(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(CARASConfig.from_env(), CARASConfig())

    def test_custom_prefix_and_overrides(self) -> None:
        values = {
            "TEST_MODE": "SLO",
            "TEST_CANDIDATE_WINDOW": "3",
            "TEST_MAX_WAIT_MS": "500",
            "TEST_AGING_TOKENS_PER_MS": "0",
            "TEST_CACHE_REUSE_WEIGHT": "0.5",
            "TEST_MIN_CACHE_BLOCKS": "0",
            "TEST_PREFILL_TOKENS_PER_MS": "2",
            "TEST_ATTENTION_CONTEXT_WEIGHT": "1",
            "TEST_ATTENTION_REFERENCE_TOKENS": "128",
            "TEST_TTFT_SLO_MS": "200",
            "TEST_PROTECT_PREEMPTED": "off",
            "TEST_METRICS_PATH": "metrics/policy.jsonl",
            "TEST_METRICS_INTERVAL": "2",
            "CACHEFLOW_MODE": "invalid-ignored-prefix",
        }
        with patch.dict(os.environ, values, clear=True):
            actual = CARASConfig.from_env(prefix="TEST_")
        self.assertEqual(
            actual,
            CARASConfig(
                mode=SchedulingMode.SLO,
                candidate_window=3,
                max_wait_ms=500,
                aging_tokens_per_ms=0,
                cache_reuse_weight=0.5,
                min_cache_blocks=0,
                prefill_tokens_per_ms=2,
                attention_context_weight=1,
                attention_reference_tokens=128,
                ttft_slo_ms=200,
                protect_preempted=False,
                metrics_path=Path("metrics/policy.jsonl"),
                metrics_interval=2,
            ),
        )

    def test_boolean_spellings(self) -> None:
        for expected, spellings in (
            (True, ("1", "true", "yes", "on")),
            (False, ("0", "false", "no", "off")),
        ):
            for raw in spellings:
                with (
                    self.subTest(raw=raw),
                    patch.dict(
                        os.environ,
                        {"CACHEFLOW_PROTECT_PREEMPTED": f" {raw.upper()} "},
                        clear=True,
                    ),
                ):
                    self.assertIs(CARASConfig.from_env().protect_preempted, expected)

    def test_invalid_environment_inputs(self) -> None:
        for name, raw in (
            ("PROTECT_PREEMPTED", "maybe"),
            ("MODE", "unknown"),
            ("CANDIDATE_WINDOW", "not-an-int"),
            ("MAX_WAIT_MS", "not-a-number"),
        ):
            with (
                self.subTest(name=name),
                patch.dict(os.environ, {f"CACHEFLOW_{name}": raw}, clear=True),
                self.assertRaises(ValueError),
            ):
                CARASConfig.from_env()

    def test_out_of_range_fields(self) -> None:
        for name, value in (
            ("candidate_window", 0),
            ("max_wait_ms", 0),
            ("aging_tokens_per_ms", -1),
            ("cache_reuse_weight", -1),
            ("min_cache_blocks", -1),
            ("prefill_tokens_per_ms", 0),
            ("attention_context_weight", -1),
            ("attention_reference_tokens", 0),
            ("ttft_slo_ms", 0),
            ("metrics_interval", 0),
        ):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                CARASConfig(**{name: value})

    def test_nonfinite_fields_rejected_directly_and_from_environment(self) -> None:
        for name in (
            "max_wait_ms",
            "aging_tokens_per_ms",
            "cache_reuse_weight",
            "prefill_tokens_per_ms",
            "attention_context_weight",
            "ttft_slo_ms",
        ):
            for value in (math.nan, math.inf, -math.inf):
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, name):
                        CARASConfig(**{name: value})
                    with (
                        patch.dict(
                            os.environ,
                            {f"CACHEFLOW_{name.upper()}": str(value)},
                            clear=True,
                        ),
                        self.assertRaisesRegex(ValueError, name),
                    ):
                        CARASConfig.from_env()


if __name__ == "__main__":
    unittest.main()
