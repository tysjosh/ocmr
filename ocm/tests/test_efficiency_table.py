"""Efficiency / systems-overhead table (paper Table V).

Pins that ``run_full_suite`` produces an ``efficiency`` section with the four
Table V columns per arm, that the text-only baseline B0 is the 0%/1.00x
reference, and that governed OCMR (B3) carries measurable context-token overhead
from the provenance/conflict annotations it adds.
"""

from __future__ import annotations

import logging

from ocm.evaluation import experiment as exp


def test_efficiency_section_shape_and_baseline():
    logging.getLogger("ocm").setLevel(logging.CRITICAL)
    report = exp.run_full_suite(
        seeds=[1337], per_category=4, stress_per_class=3, taus=(0.8, 0.9)
    )
    eff = report["efficiency"]

    # Every arm has the four Table V columns.
    for method in report["methods"]:
        cols = eff[method]
        for key in (
            "write_latency_ms",
            "query_latency_ms",
            "token_overhead_pct",
            "storage_growth_x",
        ):
            assert key in cols

    # B0 is the reference: 0% token overhead, 1.00x storage.
    assert eff["B0"]["token_overhead_pct"] == 0.0
    assert eff["B0"]["storage_growth_x"] == 1.0

    # Latencies are real, non-negative measurements.
    assert eff["B3"]["write_latency_ms"] >= 0.0
    assert eff["B3"]["query_latency_ms"] >= 0.0

    # Full OCMR adds context (provenance + conflict annotations) over the
    # text-only baseline, so its token overhead is positive.
    assert eff["B3"]["token_overhead_pct"] > 0.0
