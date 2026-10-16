"""ISIC 2019 の施設間（BCN20000 → HAM10000）で、補正前の 1 段目を属性群と潜在スライスで監査する。

audit_run.py の ISIC 版。test（`isic_test`）の log O/E の群間のばらつき（帰無分布を差し引いた SD）を、
群の切り方ごとに比べる。較正は `isic_audit_v` で行い、スライスは test 以外で決める。

- 属性：性別、65 歳で 2 区分の年齢、性別 × 年齢、性別 × 年齢 4 区分
- 部位（`anatom_site_general`、属性ではない既知の要因）
- 潜在：DINOv2 の GMM（`isic_audit` に当てる、LHCF と同じくラベル不使用）、
  `isic_audit_v` の残差で学習した監査器（h＋DINOv2、16 分位）
- 対照：ランダム群、1 段目のスコアの分位

出力: `results/isic/discovery.csv`

    uv run python -m analysis.archive.latent_bias_hypernet.isic_discovery
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.archive.latent_bias_hypernet import audit
from analysis.archive.latent_bias_hypernet.audit_corrected import logit
from analysis.archive.latent_bias_hypernet.data import PACKAGE, load

STAGE1 = "isic_bcn_s42"


def families():
    """test の各行に族ごとの群番号を振り、(族, test の Split, 較正済み確率, しきい値, 病変 id) を返す。"""
    A = load(STAGE1, "isic_audit", "dinov2")
    V = load(STAGE1, "isic_audit_v", "dinov2")
    T = load(STAGE1, "isic_test", "dinov2")
    cal = audit.Calibrator(V.base, V.target)
    pv, pt = cal(V.base), cal(T.base)
    y = T.target
    c = T.context
    sex = c.sex.to_numpy().astype(int)
    old = (c.age.fillna(50).to_numpy() >= 65).astype(int)
    age4 = np.digitize(c.age.fillna(50).to_numpy(), (40, 55, 70))
    fam = {
        "性別": sex,
        "年齢（65 歳で 2 区分）": old,
        "性別 × 年齢（4）": np.where(sex < 0, -1, sex * 2 + old),
        "性別 × 年齢 4 区分（8）": np.where(sex < 0, -1, sex * 4 + age4),
        "部位": c.anatom_site_general.to_numpy().astype(int),
        "ランダム（16）": np.random.default_rng(0).integers(0, 16, len(y)),
        "スコア分位（16、対照）": audit.quantile_slicer(V.base, 16)(T.base),
    }
    for K in (8, 16):
        fam[f"GMM {K}（DINOv2）"] = audit.GMMSlicer(A.route, K, dims=32)(T.route)
    xv, (ph, pe) = audit.auditor_inputs(V.features, V.route, V.features, V.route, dims=32)
    xt = np.concatenate([ph.transform(T.features), pe.transform(T.route)], axis=1)
    lesions = V.context.lesion_id.fillna(V.context.image).to_numpy()
    for seed in (0, 1, 2):
        r = audit.ResidualAuditor(xv, logit(pv), V.target, lesions, seed)(xt)
        fam[f"監査器 16 分位 s{seed}"] = audit.quantile_slicer(r, 16)(r)
    lesion_t = c.lesion_id.fillna(c.image).to_numpy()
    return fam, T, pt, cal.threshold, lesion_t


def main() -> None:
    fam, T, pt, threshold, lesion_t = families()
    y = T.target
    rows = []
    for name, g in fam.items():
        keep = g >= 0
        table = audit.group_metrics(g[keep], y[keep], pt[keep], threshold, 100, 10.0)
        d = audit.dispersion(table, g[keep], y[keep], pt[keep], lesion_t[keep])
        rows.append(
            {
                "family": name,
                "groups": len(table),
                "excess_sd": d["excess_sd"],
                "above_null": d["above_null"],
                "max_log_oe": table.log_oe.max(),
                "min_log_oe": table.log_oe.min(),
            }
        )
    out = pd.DataFrame(rows)
    dest = PACKAGE / "results" / "isic"
    dest.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest / "discovery.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
