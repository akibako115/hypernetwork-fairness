"""1 段目の表現・logit、振り分け係の入力 e(x)、答え合わせ用の文脈ラベルを行対応を検算して読む。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import rootutils

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

PACKAGE = ROOT / "analysis" / "archive" / "latent_bias_hypernet"
CACHE = PACKAGE / "cache"
RACE_GROUPS = ("White", "Asian", "Black", "Others")
GROUP_NAMES = tuple(f"{age}×{race}" for age in ("<65", "≥65") for race in RACE_GROUPS)


@dataclass
class Split:
    """1 split 分の入力。行順は split CSV と同じ。"""

    features: np.ndarray  # [n, 2048] float32、1 段目の分類 head 直前
    base: np.ndarray  # [n] 1 段目の logit 差（陽性 − 陰性）
    target: np.ndarray  # [n] int64
    route: np.ndarray | None  # [n, d] 振り分け係の入力
    context: pd.DataFrame  # 属性と文脈ラベル
    marked: np.ndarray  # [n] bool、植え込んだマーカー

    @property
    def group(self) -> np.ndarray:
        """年齢 2 区分 × race_group 4 区分の 8 群。"""
        ctx = self.context
        return (ctx.age.to_numpy() >= 65).astype(np.int64) * 4 + ctx.race_group.to_numpy().astype(np.int64)


def _checked(npz: np.lib.npyio.NpzFile, images: np.ndarray, what: str) -> np.lib.npyio.NpzFile:
    if not np.array_equal(npz["image"], images):
        raise ValueError(f"{what} の行順が split CSV と一致しない")
    return npz


def load(stage1: str, split: str, route: str | None = None) -> Split:
    """`stage1` の表現と、`route`（`embed/<name>` のキャッシュ名）を読む。

    `route` に `"meta"` を渡すと、モデル入力と同じ属性（年齢・性別・race・撮影条件）の one-hot を返す。
    `"h"` を渡すと、1 段目自身の表現を振り分けの入力にする。
    """
    context = pd.read_csv(CACHE / "context" / f"{split}.csv")
    images = context.image.to_numpy(dtype=str)
    s1 = _checked(np.load(CACHE / "stage1" / stage1 / f"{split}.npz"), images, f"stage1/{stage1}")
    logits = s1["logits"]
    route_x = None
    if route == "meta":
        route_x = metadata_onehot(context)
    elif route == "h":
        # 対照：1 段目自身の表現で振り分ける（スコアの層で分かれる循環が起きるかを見る）
        route_x = s1["features"].astype(np.float32)
    elif route is not None and route.startswith("h+"):
        # 1 段目の表現と e(x) を、それぞれ標準化してから連結する（モデル側と データ側の両方で分けられるように）
        emb = _checked(np.load(CACHE / "embed" / route[2:] / f"{split}.npz"), images, f"embed/{route[2:]}")
        h = s1["features"].astype(np.float32)
        e = emb["embedding"].astype(np.float32)
        route_x = np.concatenate([h / (h.std() + 1e-6), e / (e.std() + 1e-6)], axis=1)
    elif route is not None:
        emb = _checked(np.load(CACHE / "embed" / route / f"{split}.npz"), images, f"embed/{route}")
        if not np.array_equal(emb["marked"], s1["marked"]):
            raise ValueError(f"embed/{route} と stage1/{stage1} でマーカーの有無が一致しない")
        route_x = emb["embedding"].astype(np.float32)
    return Split(
        features=s1["features"].astype(np.float32),
        base=(logits[:, 1] - logits[:, 0]).astype(np.float32),
        target=s1["target"].astype(np.int64),
        route=route_x,
        context=context,
        marked=s1["marked"].astype(bool),
    )


def metadata_onehot(context: pd.DataFrame) -> np.ndarray:
    """現行の hypernet が受け取る属性に相当する入力。年齢は 10 歳刻みの one-hot にする。"""
    parts = []
    for name, card in (("sex", 2), ("race_group", 4), ("frontal_lateral", 2), ("ap_pa", 2)):
        parts.append(np.eye(card, dtype=np.float32)[context[name].to_numpy().astype(int)])
    decade = np.clip(context.age.to_numpy() // 10, 1, 9).astype(int) - 1
    parts.append(np.eye(9, dtype=np.float32)[decade])
    return np.concatenate(parts, axis=1)


def subset(s: Split, mask: np.ndarray) -> Split:
    """行の部分集合。"""
    return Split(
        features=s.features[mask],
        base=s.base[mask],
        target=s.target[mask],
        route=s.route[mask] if s.route is not None else None,
        context=s.context[mask].reset_index(drop=True),
        marked=s.marked[mask],
    )


def holdout_by_patient(s: Split, frac: float, seed: int = 0) -> tuple[Split, Split]:
    """患者単位で `frac` を取り分け、(残り, 取り分けた側) を返す。"""
    patients = s.context.image.str.split("/").str[1]
    uniq = np.sort(patients.unique())
    held = set(np.random.default_rng(seed).permutation(uniq)[: int(len(uniq) * frac)])
    mask = patients.isin(held).to_numpy()
    return subset(s, ~mask), subset(s, mask)
