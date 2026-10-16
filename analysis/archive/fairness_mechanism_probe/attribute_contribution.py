"""画像経路の属性依存：M01 の分類器を固定し、表現から属性の線形部分空間を LEACE で除いたときの logit 変化。

問い：metadata を受け取らない ERM でも、画像に符号化された属性が判断にどれだけ入っているか。
hypernet の反実仮想 metadata 解析（明示経路）と対になる、画像経路の測定。
出力: `results/erasure/attribute_logit_contribution.csv`（群ごとの平均 Δlogit = 元 − 消去後）

    uv run python analysis/archive/fairness_mechanism_probe/attribute_contribution.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from erasure import leace
from features import GROUP_NAMES, M01_RUNS, PACKAGE, load_split
from sklearn.metrics import roc_auc_score


def classifier_direction(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """best checkpoint の fc から、s = z1 − z0 を与える重みと切片。"""
    path = next((PACKAGE / "runs" / M01_RUNS[seed] / "checkpoints").glob("best*.ckpt"))
    state = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    weight, bias = state["net.fc.weight"].float(), state["net.fc.bias"].float()
    return weight[1] - weight[0], bias[1] - bias[0]


def main() -> None:
    rows = []
    for seed in M01_RUNS:
        w, c = classifier_direction(seed)
        train, test = load_split(seed, "train"), load_split(seed, "test")
        x_train, x_test = torch.from_numpy(train.features).cuda(), torch.from_numpy(test.features).cuda()
        score = (x_test.cpu() @ w + c).numpy()
        if not np.allclose(score, test.logits[:, 1] - test.logits[:, 0], atol=0.05):
            raise ValueError("fc から再計算した logit が予測 cache と一致しない")
        decade = train.frame.age.floordiv(10).clip(1, 9).to_numpy().astype(int) - 1
        concepts = {
            "age_linear": torch.from_numpy(((train.frame.age.to_numpy() - 60) / 18).astype(np.float32))[:, None],
            "age_decade": torch.from_numpy(np.eye(9, dtype=np.float32)[decade]),
            "race": torch.nn.functional.one_hot(torch.from_numpy(train.frame.race_group.to_numpy()), 4).float(),
            "sex": torch.from_numpy(train.frame.sex.to_numpy().astype(np.float32))[:, None],
        }
        old = test.frame.age.to_numpy() >= 65
        for name, z in concepts.items():
            erased = (leace(x_train, z.cuda())(x_test).cpu() @ w + c).numpy()
            delta = score - erased
            row = {
                "seed": seed,
                "concept": name,
                "auc_full": roc_auc_score(test.target, score),
                "auc_erased": roc_auc_score(test.target, erased),
            }
            row.update({n: delta[test.group == k].mean() for k, n in enumerate(GROUP_NAMES)})
            row["old_auc_erased"] = roc_auc_score(test.target[old], erased[old])
            row["young_auc_erased"] = roc_auc_score(test.target[~old], erased[~old])
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(PACKAGE / "results" / "erasure" / "attribute_logit_contribution.csv", index=False)
    print(table.groupby("concept").mean(numeric_only=True).drop(columns="seed").round(3).T.to_string())


if __name__ == "__main__":
    main()
