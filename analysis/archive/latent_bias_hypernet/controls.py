"""対照：e(x) を「グループ分けの手がかり」ではなく、そのまま追加の予測変数として足した場合。

    補正後の logit = base(x) + w · PCA(e(x)) + b   （fit split で学習、val で正則化を選ぶ）

振り分け係（K 個の切片）の改善が、この連続な追加より小さいなら、コードは e(x) の情報の一部を
量子化して使っているだけと読める。出力: `results/controls.csv`

    uv run python -m analysis.archive.latent_bias_hypernet.controls --stage1 a_s42 --route raddino
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from analysis.archive.latent_bias_hypernet.data import PACKAGE, load
from analysis.archive.latent_bias_hypernet.latent_codes import performance, weighted_bce


def fit_offset(x: np.ndarray, base: np.ndarray, y: np.ndarray, pos_rate: float, wd: float, device: str):
    """base を offset に固定した class weight 付きロジスティック回帰。"""
    X = torch.as_tensor(x, device=device)
    B = torch.as_tensor(base, device=device)
    Y = torch.as_tensor(y, device=device)
    w = torch.zeros(x.shape[1], device=device, requires_grad=True)
    b = torch.zeros(1, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = weighted_bce(B + X @ w + b, Y, pos_rate) + wd * (w**2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach(), b.detach()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1", default="a_s42")
    parser.add_argument("--route", default="raddino")
    parser.add_argument("--fit", default="train_b")
    parser.add_argument("--dims", type=int, nargs="+", default=[16, 64, 256])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    fit, val, test = (load(args.stage1, s, args.route) for s in (args.fit, "val", "test"))
    pos_rate = float(fit.target.mean())
    rows = []
    for dims in args.dims:
        pca = PCA(dims, random_state=0).fit(fit.route)
        z = {k: pca.transform(s.route).astype(np.float32) for k, s in (("fit", fit), ("val", val), ("test", test))}
        sd = z["fit"].std(0) + 1e-6
        z = {k: v / sd for k, v in z.items()}
        best = None
        for wd in (1e-5, 1e-4, 1e-3, 1e-2):
            w, b = fit_offset(z["fit"], fit.base, fit.target, pos_rate, wd, args.device)
            score_val = val.base + (torch.as_tensor(z["val"], device=args.device) @ w + b).cpu().numpy()
            val_loss = float(weighted_bce(torch.as_tensor(score_val), torch.as_tensor(val.target), pos_rate))
            if best is None or val_loss < best[0]:
                best = (val_loss, wd, w, b)
        _, wd, w, b = best
        score = test.base + (torch.as_tensor(z["test"], device=args.device) @ w + b).cpu().numpy()
        base_perf, perf = performance(test.base, test, pos_rate), performance(score, test, pos_rate)
        row = {"stage1": args.stage1, "route": args.route, "fit": args.fit, "pca": dims, "wd": wd}
        row.update({f"d_{k}": perf[k] - base_perf[k] for k in perf})
        rows.append(row)
        print(row, flush=True)
    out = PACKAGE / "results" / "controls.csv"
    table = pd.DataFrame(rows)
    if out.exists():
        table = pd.concat([pd.read_csv(out), table]).drop_duplicates(["stage1", "route", "fit", "pca"], keep="last")
    table.to_csv(out, index=False)


if __name__ == "__main__":
    main()
