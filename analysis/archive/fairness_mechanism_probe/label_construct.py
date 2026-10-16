"""ラベルの構成概念：陽性（No Finding）を、レポートが正常を明言したものと、そうでないものに分けて年齢差を比べる。

問い：高齢群の低い AUROC は、画像の判別の難しさか、「No Finding」というラベルの意味の揺れか。
CheXpert Plus の impression を正規表現で分類する（臨床的な正解ではなく、レポートの書かれ方の分類）。

- affirmed_normal：「no acute cardiopulmonary」「normal chest」「lungs are clear」などを含む
- focused_followup：デバイス・気胸・前回比較など特定の問いに答えるだけで、正常の明言がない
- other：それ以外

M01 3seed の test logit 平均を使う。出力: `results/label_construct/auroc_by_report_type.csv`

    uv run python analysis/archive/fairness_mechanism_probe/label_construct.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from features import GROUP_NAMES, PACKAGE, REPOSITORY_ROOT, load_split
from sklearn.metrics import roc_auc_score

AFFIRMED = "|".join(
    [
        r"no acute cardiopulmonary",
        r"no acute (?:intrathoracic|chest|disease|process|abnormal)",
        r"normal chest",
        r"normal (?:radiograph|study|exam)",
        r"lungs are clear",
        r"clear lungs",
        r"no active (?:cardiopulmonary|disease)",
        r"unremarkable",
    ]
)
FOCUSED = "|".join(
    [
        r"pneumothorax|catheter|picc|\bline\b|tube|swan|drain",
        r"interval|removal|compared|no (?:significant )?change|unchanged|stable",
    ]
)


def report_type(impression: pd.Series) -> pd.Series:
    text = impression.fillna("").str.lower()
    affirmed = text.str.contains(AFFIRMED)
    focused = text.str.contains(FOCUSED) & ~affirmed
    return pd.Series(
        np.select([affirmed, focused], ["affirmed_normal", "focused_followup"], "other"), index=impression.index
    )


def main() -> None:
    tests = [load_split(seed, "test") for seed in (42, 43, 44)]
    frame = tests[0].frame.copy()
    frame["score"] = np.mean([t.logits[:, 1] - t.logits[:, 0] for t in tests], axis=0)
    frame["group"] = tests[0].group
    plus = pd.read_csv(
        REPOSITORY_ROOT / "data/chexpert/df_chexpert_plus_240401.csv", usecols=["path_to_image", "section_impression"]
    )
    frame = frame.merge(plus.rename(columns={"path_to_image": "image"}), on="image", how="left", validate="one_to_one")
    frame["report_type"] = report_type(frame.section_impression)
    pos = frame.target == 1
    print("陽性のレポート種別（年齢別の割合）")
    print(pd.crosstab(frame.age[pos] >= 65, frame.report_type[pos], normalize="index").round(3))
    rows = []
    bands = {
        "<50": frame.age < 50,
        "50-64": (frame.age >= 50) & (frame.age < 65),
        "65-79": (frame.age >= 65) & (frame.age < 80),
        "80+": frame.age >= 80,
    }
    for kind in ("all", "affirmed_normal", "focused_followup", "other"):
        keep_pos = pos if kind == "all" else pos & (frame.report_type == kind)
        row = {"positives": kind}
        for band, m in bands.items():
            sel = m & (keep_pos | ~pos)
            row[f"auc {band}"] = roc_auc_score(frame.target[sel], frame.score[sel])
            row[f"n_pos {band}"] = int((keep_pos & m).sum())
            row[f"mean_score_pos {band}"] = frame.score[keep_pos & m].mean()
        for g, name in enumerate(GROUP_NAMES):
            sel = (frame.group == g) & (keep_pos | ~pos)
            row[name] = (
                roc_auc_score(frame.target[sel], frame.score[sel])
                if (keep_pos & (frame.group == g)).sum() >= 10
                else np.nan
            )
        rows.append(row)
    table = pd.DataFrame(rows)
    out = PACKAGE / "results" / "label_construct"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "auroc_by_report_type.csv", index=False)
    pd.set_option("display.width", 250)
    print(table.round(3).T.to_string())


if __name__ == "__main__":
    main()
