"""Fail CI when a candidate run regresses latency, cost, or quality."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    """Compare two evaluation reports against configurable thresholds."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)

    # Defaults allow ordinary run-to-run noise without hiding material regressions.
    ap.add_argument("--latency_p95_regress_pct", type=float, default=15.0)
    ap.add_argument("--latency_p99_regress_pct", type=float, default=20.0)
    ap.add_argument("--tokens_avg_regress_pct", type=float, default=15.0)
    ap.add_argument("--quality_avg_min", type=float, default=0.80)  # avg embedding cosine vs baseline
    ap.add_argument("--quality_p10_min", type=float, default=0.65)

    args = ap.parse_args()

    b = json.loads(Path(args.baseline).read_text())
    c = json.loads(Path(args.candidate).read_text())

    fails: list[str] = []

    def regress_pct(bv: float, cv: float) -> float:
        if bv <= 0:
            return 0.0 if cv <= 0 else 999.0
        return ((cv - bv) / bv) * 100.0

    b_p95 = float(b["latency_ms"]["p95"])
    c_p95 = float(c["latency_ms"]["p95"])
    if regress_pct(b_p95, c_p95) > args.latency_p95_regress_pct:
        fails.append(f"Latency p95 regressed: baseline {b_p95:.1f}ms → candidate {c_p95:.1f}ms")

    b_p99 = float(b["latency_ms"]["p99"])
    c_p99 = float(c["latency_ms"]["p99"])
    if regress_pct(b_p99, c_p99) > args.latency_p99_regress_pct:
        fails.append(f"Latency p99 regressed: baseline {b_p99:.1f}ms → candidate {c_p99:.1f}ms")

    b_tok = float(b["tokens_proxy"]["avg"])
    c_tok = float(c["tokens_proxy"]["avg"])
    if regress_pct(b_tok, c_tok) > args.tokens_avg_regress_pct:
        fails.append(f"Tokens proxy avg regressed: baseline {b_tok:.1f} → candidate {c_tok:.1f}")

    # A baseline-only report has no similarity scores, so quality gates are optional.
    qavg = c.get("quality_similarity_vs_baseline", {}).get("avg")
    qp10 = c.get("quality_similarity_vs_baseline", {}).get("p10")

    if qavg is not None and float(qavg) < args.quality_avg_min:
        fails.append(f"Quality avg similarity too low: {float(qavg):.3f} < {args.quality_avg_min:.3f}")
    if qp10 is not None and float(qp10) < args.quality_p10_min:
        fails.append(f"Quality p10 similarity too low: {float(qp10):.3f} < {args.quality_p10_min:.3f}")

    if fails:
        print("Regression gates FAILED:\n")
        for f in fails:
            print(f"- {f}")
        sys.exit(1)

    print(" Regression gates PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()
