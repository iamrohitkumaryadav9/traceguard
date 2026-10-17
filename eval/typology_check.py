"""Are the structural typology flags informative? Uses labels, so it lives in eval/ only.

Uses labeled nodes in time steps 1-34 ONLY (never test). Run: python -m eval.typology_check
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.tools.context import CHAIN_MIN, get_context

OUT = Path("eval/results/typology_check.md")


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def table(ctx, mask_rows, label):
    idx = np.where(mask_rows)[0]
    fan_in, fan_out, chain = ctx.pattern_flags(idx)
    y = ctx._y[idx] == 1
    flags = {"fan_in": fan_in, "fan_out": fan_out, "chain": chain, "any": fan_in | fan_out | chain}
    rows = []
    for name, f in flags.items():
        for state, m in (("flagged", f), ("unflagged", ~f)):
            n, k = int(m.sum()), int(y[m].sum())
            lo, hi = wilson(k, n)
            rows.append({"steps": label, "pattern": name, "group": state, "n": n, "illicit": k,
                         "illicit_rate": k / n if n else np.nan, "ci95_low": lo, "ci95_high": hi})
    return pd.DataFrame(rows), y.mean()


def md(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines)


def main():
    ctx = get_context()
    labeled = ctx._y >= 0
    parts, base = [], {}
    for label, (a, b) in (("1-34", (1, 34)), ("1-29", (1, 29)), ("30-34", (30, 34))):
        m = labeled & (ctx.time_step >= a) & (ctx.time_step <= b)
        df, rate = table(ctx, m, label)
        parts.append(df)
        base[label] = rate
    res = pd.concat(parts, ignore_index=True)
    main_t = res[res.steps == "1-34"].pivot(index="pattern", columns="group", values="illicit_rate")
    main_t["lift"] = main_t["flagged"] / main_t["unflagged"]
    print(f"Fan thresholds (99th pct of degree, steps 1-29, min 3): in >= {ctx.fan_in_thr}, out >= {ctx.fan_out_thr}; "
          f"chain >= {CHAIN_MIN}")
    print(f"Base illicit rate among labeled nodes, steps 1-34: {base['1-34']:.4f}")
    pd.set_option("display.width", 200)
    print(res[res.steps == "1-34"].round(4).to_string(index=False))
    print("\nLift (flagged rate / unflagged rate), steps 1-34:")
    print(main_t.round(3).to_string())

    lines = [
        "# Typology flag check", "",
        "Illicit rate of flagged vs unflagged LABELED nodes. Time steps 1-34 only; the test period is never used.",
        f"Fan thresholds: in-degree >= {ctx.fan_in_thr}, out-degree >= {ctx.fan_out_thr} (99th percentile over "
        f"steps 1-29, minimum 3). Chain: chain_len >= {CHAIN_MIN}. 'any' = at least one flag.",
        "Elliptic has no typology labels, so this only asks whether the structural flags carry signal about the "
        "illicit label, not whether they match real typologies.", "",
        f"Base illicit rate (labeled, steps 1-34): {base['1-34']:.4f}.", "",
        md(res), "", "## Lift, steps 1-34", "", md(main_t.reset_index()), "",
    ]
    OUT.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
