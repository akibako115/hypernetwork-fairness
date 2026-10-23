"""学習目的関数が LitModule と交わす約束を定義する。"""

import lightning as L
import torch.nn as nn


class Objective(nn.Module):
    """全ての学習目的関数の基底。LitModule はこの約束だけを通して目的関数を扱う。

    forward は `ObjectiveInput` からスカラー損失を返す。損失の計算の外の時点で状態を動かす目的関数
    （双対変数の確定、epoch 末の全件評価、別に引いたバッチの推論など）は、その処理を Lightning の
    callback として `callbacks()` で返す。LitModule はそれを Trainer に渡すだけで、手法ごとの処理を持たない。
    `uses_held_out` は、DataModule の held-out を使う目的関数だけが True にする。
    """

    uses_held_out: bool = False

    def callbacks(self) -> list[L.Callback]:
        """この目的関数が学習中に必要とする callback を返す。

        Args:
            なし

        Returns:
            list[Callback]: 毎回新しく作った callback。損失の計算だけで完結する目的関数は空。
        """
        return []
