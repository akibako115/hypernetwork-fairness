"""seed だけを変えた条件どうしで、test のグループ割り当てがどれだけ一致するか（ARI の平均）。

name の末尾 `_s<seed>` を除いた部分が同じ条件を 1 組とみなす。出力: `results/stability.csv`

    uv run python -m analysis.archive.latent_bias_hypernet.stability dv_a_s42
"""

from __future__ import annotations

import itertools
import re
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from analysis.archive.latent_bias_hypernet.data import PACKAGE


def main() -> None:
    prefix = sys.argv[1] if len(sys.argv) > 1 else ""
    assign_dir = PACKAGE / "cache" / "assign"
    groups: dict[str, list[np.ndarray]] = {}
    for path in sorted(assign_dir.glob(f"{prefix}*_test.npy")):
        name = path.name.removesuffix("_test.npy")
        key = re.sub(r"_s\d+$", "", name)
        groups.setdefault(key, []).append(np.load(path))
    rows = []
    for key, assigns in groups.items():
        if len(assigns) < 2:
            continue
        aris = [adjusted_rand_score(a, b) for a, b in itertools.combinations(assigns, 2)]
        rows.append({"condition": key, "seeds": len(assigns), "ari_mean": np.mean(aris), "ari_min": np.min(aris)})
    table = pd.DataFrame(rows)
    out = PACKAGE / "results" / "stability.csv"
    if out.exists():
        table = pd.concat([pd.read_csv(out), table]).drop_duplicates("condition", keep="last")
    table.to_csv(out, index=False)
    print(table[table.condition.str.startswith(prefix)].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
