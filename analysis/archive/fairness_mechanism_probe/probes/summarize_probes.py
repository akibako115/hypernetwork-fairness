"""fine-tune probe の log.json から、選択規則ごとの test 群別 AUROC を表にする。

選択規則は「val 全体 AUROC 最大」と「val 高齢 AUROC 最大」の 2 つ。どの probe も同じ規則で比べる。
出力: `results/probes/probe_summary.csv`

    uv run python analysis/archive/fairness_mechanism_probe/probes/summarize_probes.py
"""

from __future__ import annotations

import json

import pandas as pd
import rootutils

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
PACKAGE = ROOT / "analysis" / "archive" / "fairness_mechanism_probe"
GROUPS = ("<65×White", "<65×Asian", "<65×Black", "<65×Others", "≥65×White", "≥65×Asian", "≥65×Black", "≥65×Others")


def main() -> None:
    rows = []
    for log_path in sorted((PACKAGE / "cache" / "probes").glob("*/log.json")):
        log = json.loads(log_path.read_text())
        for rule in ("overall", "old"):
            best = max(log["epochs"], key=lambda e: e["val"][rule])
            test = best["test"]
            row = {
                "probe": log_path.parent.name,
                "selected_by_val": rule,
                "epoch": best["epoch"],
                "n_train": log["n_train"],
            }
            row.update({k: test[k] for k in ("overall", "young", "old")})
            row.update({name: test[f"g{k}"] for k, name in enumerate(GROUPS)})
            rows.append(row)
    table = pd.DataFrame(rows)
    out = PACKAGE / "results" / "probes"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "probe_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
