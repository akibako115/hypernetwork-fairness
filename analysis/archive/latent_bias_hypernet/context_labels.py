"""発見手法の答え合わせに使う「モデルに渡していない要因」を、全 split の画像ごとに作る。

- `report_type`：impression の書かれ方（affirmed_normal / focused_followup / other）。
  正規表現は `fairness_mechanism_probe/label_construct.py` と同じ。臨床的な正誤ではない。
- `followup`：同じ患者の 2 回目以降のレポート（`patient_report_date_order > 1`）
- `device`：impression がデバイス（line・tube・catheter など）に触れる
- `pacer`：レポート全文がペースメーカー・植込み型除細動器に触れる（植え込み実験 2 の部分集団）
- `picc`・`central_line`・`sternotomy`・`surgical_clip`：レポート全文がそれぞれに触れる（用量反応の植え込み。
  画像に写り属性には無い要因で、年齢・性別への偏り方が違う）

属性（age・sex・race_group・frontal_lateral・ap_pa・insurance_type）は split CSV の列をそのまま残す。

出力: `cache/context/<split>.csv`（split CSV と同じ行順）

    uv run python analysis/archive/latent_bias_hypernet/context_labels.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import rootutils

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

PACKAGE = ROOT / "analysis" / "archive" / "latent_bias_hypernet"
SPLITS = ROOT / "data" / "chexpert" / "splits_crossfit"
KEEP = ("image", "target", "age", "sex", "race_group", "frontal_lateral", "ap_pa", "insurance_type")
# fairness_mechanism_probe/label_construct.py と同じ正規表現。あちらは script 実行前提で import できないので複製する。
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
PACER = r"pacemaker|pacer|aicd|defibrillator"
REPORT_FACTORS = {
    "picc": r"\bpicc\b",
    "central_line": r"central venous|\bcvc\b|internal jugular|subclavian (?:catheter|line)|\bij\b",
    "sternotomy": r"sternotomy|sternal wire|cabg|bypass graft",
    "surgical_clip": r"surgical clip|clips",
}
DEVICE = r"catheter|picc|\bline\b|tube|swan|drain|pacemaker|pacer|wire|clip"


def report_type(impression: pd.Series) -> pd.Series:
    text = impression.fillna("").str.lower()
    affirmed = text.str.contains(AFFIRMED)
    focused = text.str.contains(FOCUSED) & ~affirmed
    return pd.Series(
        np.select([affirmed, focused], ["affirmed_normal", "focused_followup"], "other"), index=impression.index
    )


def main() -> None:
    plus = pd.read_csv(
        ROOT / "data/chexpert/df_chexpert_plus_240401.csv",
        usecols=["path_to_image", "section_impression", "report", "patient_report_date_order"],
    ).rename(columns={"path_to_image": "image"})
    out = PACKAGE / "cache" / "context"
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train_a", "train_b", "val", "test"):
        frame = pd.read_csv(SPLITS / f"{split}.csv")[list(KEEP)]
        merged = frame.merge(plus, on="image", how="left", validate="one_to_one")
        if len(merged) != len(frame) or merged.section_impression.isna().mean() > 0.05:
            raise ValueError(f"{split}: CheXpert Plus との対応が取れていない")
        text = merged.section_impression.fillna("").str.lower()
        merged["report_type"] = report_type(merged.section_impression)
        merged["followup"] = merged.patient_report_date_order > 1
        merged["device"] = text.str.contains(DEVICE)
        report = merged.report.fillna("").str.lower()
        merged["pacer"] = report.str.contains(PACER)
        for name, pattern in REPORT_FACTORS.items():
            merged[name] = report.str.contains(pattern)
        merged.drop(columns=["section_impression", "report"]).to_csv(out / f"{split}.csv", index=False)
        pos = merged.target == 1
        print(
            split,
            len(merged),
            "陽性の report_type:",
            merged.report_type[pos].value_counts(normalize=True).round(3).to_dict(),
        )


if __name__ == "__main__":
    main()
