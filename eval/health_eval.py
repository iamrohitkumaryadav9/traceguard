"""Evaluate check_model_health on test steps 35-49 (constants are fixed in src/tools/health.py).

Run: python -m eval.health_eval
"""
from pathlib import Path

import pandas as pd

from src.split import SHUTDOWN_STEP, TEST_STEPS
from src.tools import run_tool

OUT = Path("eval/results/model_health.md")


def main():
    rows = [run_tool("check_model_health", {"time_step": t}) for t in range(TEST_STEPS[0], TEST_STEPS[1] + 1)]
    df = pd.DataFrame(rows)[["time_step", "status", "window_steps", "window_recall", "reference_recall",
                             "n_illicit_in_window", "lag"]]
    df["window_steps"] = df["window_steps"].map(lambda s: f"{s[0]}-{s[-1]}" if s else "-")
    pd.set_option("display.width", 200)
    print(df.round(4).to_string(index=False))
    degraded = df.loc[df.status == "degraded", "time_step"].tolist()
    first = degraded[0] if degraded else None
    false_alarms = [t for t in degraded if t < SHUTDOWN_STEP]
    print(f"\nFirst degraded step: {first} (shutdown at {SHUTDOWN_STEP}; detection delay "
          f"{None if first is None else first - SHUTDOWN_STEP} steps)")
    print(f"Degraded flags in steps 35-{SHUTDOWN_STEP - 1} (false alarms): {false_alarms or 'none'}")
    print(f"Steps with insufficient data: {df.loc[df.status == 'insufficient_data', 'time_step'].tolist() or 'none'}")

    lines = ["# Model health monitor (delayed labels), test steps 35-49", "",
             "Constants fixed in advance: lag 2, window 3 steps ending at time_step - 2, pooled recall at the alert "
             "threshold, reference = validation steps 30-34, degraded if window recall < 0.5 x reference, at least "
             "20 illicit nodes required. The constants were designed after the post-shutdown failure was observed.", ""]
    cols = list(df.columns)
    lines += ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in r) + " |")
    lines += ["", f"First degraded step: {first}. False alarms in 35-{SHUTDOWN_STEP - 1}: {false_alarms or 'none'}."]
    OUT.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
