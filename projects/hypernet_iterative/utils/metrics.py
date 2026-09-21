"""属性グループ別の公平性指標と accuracy を計算する関数群。"""

from collections.abc import Mapping, Sequence

import torch
from torchmetrics.functional.classification import binary_auroc


def _validate_attribute_tensor(values: torch.Tensor, batch_size: int, key: str) -> torch.Tensor:
    """属性テンソルが `[batch_size, num_attributes]` であることを検証する。"""
    if values.ndim != 2:
        raise ValueError(f"{key} must have shape [batch_size, num_attributes]")
    if values.size(0) != batch_size:
        raise ValueError(f"{key} batch size ({values.size(0)}) does not match targets batch size ({batch_size})")
    return values


def _resolve_missing_mask(
    values: torch.Tensor,
    missing: torch.Tensor | None,
) -> torch.Tensor:
    """属性ごとの欠損マスクを返す。値が負のとき欠損とみなす。"""
    if missing is not None:
        missing = _validate_attribute_tensor(missing, values.size(0), "categorical_missing")
        return missing.to(dtype=torch.bool, device=values.device)
    return values < 0


def _resolve_attribute_names(
    key: str,
    width: int,
    attribute_names: Mapping[str, Sequence[str]] | None,
) -> list[str]:
    """属性名を解決する。"""
    names = None if attribute_names is None else attribute_names.get(key)
    if names is None:
        return [f"{key}[{index}]" for index in range(width)]
    if len(names) != width:
        raise ValueError(f"attribute_names[{key!r}] must contain {width} names, but got {len(names)}")
    return list(names)


def build_eval_attributes(
    categorical: torch.Tensor,
    attribute_names: Sequence[str] | None = None,
    categorical_missing: torch.Tensor | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, list[str]]]:
    """
    epoch バッファから評価用のグループ属性を構築する。

    全属性がカテゴリカル（バイナリ済み）として扱われる。
    値が -1 のサンプルは欠損として集計から除外される。

    Args:
        categorical:     shape `[N, num_cat]`。各列が1属性に対応。
        attribute_names: 列ごとの属性名。None の場合は "attr[i]" 形式。

    Returns:
        (eval_attributes, eval_attribute_names) のタプル。
        compute_metrics_by_attribute にそのまま渡せる形式。
    """
    num_attrs = categorical.shape[1]
    if attribute_names is None:
        names = [f"attr[{i}]" for i in range(num_attrs)]
    else:
        names = list(attribute_names)
    attributes = {"categorical": categorical.long()}
    if categorical_missing is not None:
        attributes["categorical_missing"] = categorical_missing.bool()
    return attributes, {"categorical": names}


def compute_fairness_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    attributes: Mapping[str, torch.Tensor] | None,
    attribute_names: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, dict[str, float]]:
    """
    属性グループ間の fairness 指標 (Eopp0, Eopp1, Eodds) を計算する。

    各指標はクラスごとに one-vs-rest で算出し、macro 平均をとる。
    - Eopp1: グループ間 TPR (True Positive Rate) の最大差のクラス平均
    - Eopp0: グループ間 TNR (True Negative Rate) の最大差のクラス平均
    - Eodds: クラスごとの (TPR gap + FPR gap) / 2 の macro 平均

    Args:
        logits: モデルの出力。shape は `[N, num_classes]` を想定。
        targets: 正解ラベル。shape は `[N]` を想定。
        attributes: 属性辞書。`categorical` と任意の `categorical_missing` を受け付ける。
        attribute_names: 属性列名。未指定なら `categorical[0]` のような名前を使う。

    Returns:
        例:
        {
            "sex": {"Eopp0": 0.05, "Eopp1": 0.08, "Eodds": 0.065},
            "age_group": {"Eopp0": 0.03, "Eopp1": 0.10, "Eodds": 0.065},
        }
    """
    if attributes is None:
        return {}

    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch_size, num_classes]")

    # targetの変換・バリデーションチェック
    targets = targets.view(-1).long()
    batch_size = targets.size(0)
    if logits.size(0) != batch_size:
        raise ValueError(f"logits batch size ({logits.size(0)}) does not match targets batch size ({batch_size})")

    # クラスの数を取得
    num_classes = logits.size(1)
    preds = torch.argmax(logits, dim=1)

    # 属性の取得・バリデーションチェック
    values = attributes.get("categorical")
    if values is None:
        return {}

    # 結果の初期化
    result: dict[str, dict[str, float]] = {}

    # 属性のバリデーション・欠損マスクの取得・属性名の取得
    values = _validate_attribute_tensor(values, batch_size, "categorical")
    missing = _resolve_missing_mask(values, attributes.get("categorical_missing"))
    names = _resolve_attribute_names("categorical", values.size(1), attribute_names)

    # 属性ごとに公平性指標を計算
    for index, name in enumerate(names):
        column = values[:, index]
        valid_mask = ~missing[:, index]

        # 属性値の欠損値を除外した一意な値を取得
        groups = torch.unique(column[valid_mask], sorted=True)
        if groups.numel() < 2:
            result[name] = {}
            continue

        if num_classes == 2:
            # 2クラス: positive class (class 1) に対して直接 TPR/FPR/TNR gap を計算
            tprs: list[float] = []
            fprs: list[float] = []
            tnrs: list[float] = []
            # グループごとに正例/負例マスクを作り、TPR（正例中の正解率）・
            # FPR/TNR（負例中の誤答率/正解率）を求める
            for g in groups:
                group_mask = valid_mask & column.eq(g)
                pos = group_mask & targets.eq(1)
                neg = group_mask & targets.eq(0)
                if pos.any():
                    tprs.append(float(preds[pos].eq(1).float().mean().item()))
                if neg.any():
                    fprs.append(float(preds[neg].eq(1).float().mean().item()))
                    tnrs.append(float(preds[neg].eq(0).float().mean().item()))

            # グループ間の最大差（gap）から Eopp1 / Eopp0 / Eodds を求める
            eopp1 = (max(tprs) - min(tprs)) if len(tprs) >= 2 else float("nan")
            eopp0 = (max(tnrs) - min(tnrs)) if len(tnrs) >= 2 else float("nan")
            fpr_gap = (max(fprs) - min(fprs)) if len(fprs) >= 2 else float("nan")
            eodds = (eopp1 + fpr_gap) / 2 if eopp1 == eopp1 and fpr_gap == fpr_gap else float("nan")
        else:
            # 多クラス: one-vs-rest でクラスごとに TPR/FPR/TNR gap を計算し macro 平均
            tpr_gaps: list[float] = []
            tnr_gaps: list[float] = []
            eodds_gaps: list[float] = []

            for c in range(num_classes):
                class_tprs: list[float] = []
                class_fprs: list[float] = []
                class_tnrs: list[float] = []

                # クラス c を positive とみなし（one-vs-rest）、グループごとに
                # 正例/負例マスクを作って TPR・FPR・TNR を求める
                for g in groups:
                    group_mask = valid_mask & column.eq(g)
                    pos = group_mask & targets.eq(c)
                    neg = group_mask & targets.ne(c)
                    if pos.any():
                        class_tprs.append(float(preds[pos].eq(c).float().mean().item()))
                    if neg.any():
                        class_fprs.append(float(preds[neg].eq(c).float().mean().item()))
                        class_tnrs.append(float(preds[neg].ne(c).float().mean().item()))

                # クラス c におけるグループ間の最大差（gap）を求める
                tpr_gap = (max(class_tprs) - min(class_tprs)) if len(class_tprs) >= 2 else float("nan")
                fpr_gap = (max(class_fprs) - min(class_fprs)) if len(class_fprs) >= 2 else float("nan")
                tnr_gap = (max(class_tnrs) - min(class_tnrs)) if len(class_tnrs) >= 2 else float("nan")
                if tpr_gap == tpr_gap:
                    tpr_gaps.append(tpr_gap)
                if tnr_gap == tnr_gap:
                    tnr_gaps.append(tnr_gap)
                if tpr_gap == tpr_gap and fpr_gap == fpr_gap:
                    eodds_gaps.append((tpr_gap + fpr_gap) / 2)

            # クラスごとの gap を macro 平均して Eopp1 / Eopp0 / Eodds を求める
            eopp1 = sum(tpr_gaps) / len(tpr_gaps) if tpr_gaps else float("nan")
            eopp0 = sum(tnr_gaps) / len(tnr_gaps) if tnr_gaps else float("nan")
            eodds = sum(eodds_gaps) / len(eodds_gaps) if eodds_gaps else float("nan")

        result[name] = {"Eopp0": eopp0, "Eopp1": eopp1, "Eodds": eodds}

    if num_classes == 2:
        group_performance = _compute_group_performance_metrics(logits, targets, attributes, attribute_names)
        for name, metrics in group_performance.items():
            result[name].update(
                {
                    "auroc_gap": metrics["auroc_gap"],
                    "bacc_gap": metrics["bacc_gap"],
                    "worst_group_auroc": metrics["worst_group_auroc"],
                    "worst_group_bacc": metrics["worst_group_bacc"],
                }
            )

    return result


def compute_metrics_by_attribute(
    logits: torch.Tensor,
    targets: torch.Tensor,
    attributes: Mapping[str, torch.Tensor] | None,
    attribute_names: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, dict[int, float]]:
    """
    属性値ごとに accuracy を集計する。

    Args:
        logits: モデルの出力。shape は `[batch_size, num_classes]` を想定。
        targets: 正解ラベル。shape は `[batch_size]` を想定。
        attributes: 属性辞書。`categorical` と任意の `categorical_missing` を受け付ける。
        attribute_names: 属性列名。未指定なら `categorical[0]` のような名前を使う。

    Returns:
        例:
        {
            "sex": {0: 0.75, 1: 0.80},
            "age_group": {0: 1.0, 1: 0.85, 2: 0.70},
        }
    """
    if attributes is None:
        return {}

    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch_size, num_classes]")

    # targetの変換・バリデーションチェック
    targets = targets.view(-1).long()
    batch_size = targets.size(0)
    if logits.size(0) != batch_size:
        raise ValueError(f"logits batch size ({logits.size(0)}) does not match targets batch size ({batch_size})")

    # 予測クラスを求め、正解ラベルと一致するかのマスクを作る
    preds = torch.argmax(logits, dim=1)
    correct = preds.eq(targets)

    metrics_by_attribute: dict[str, dict[int, float]] = {}
    values = attributes.get("categorical")
    if values is None:
        return {}

    # 属性のバリデーション・欠損マスクの取得・属性名の取得
    values = _validate_attribute_tensor(values, batch_size, "categorical")
    missing = _resolve_missing_mask(values, attributes.get("categorical_missing"))
    names = _resolve_attribute_names("categorical", values.size(1), attribute_names)

    # 属性ごとに accuracy を集計する
    for index, name in enumerate(names):
        column = values[:, index]
        valid_mask = ~missing[:, index]
        if not torch.any(valid_mask):
            metrics_by_attribute[name] = {}
            continue

        valid_values = column[valid_mask]
        group_accs: dict[int, float] = {}

        # 属性値ごとにサンプルを絞り込み、accuracy を集計する
        for group_value in torch.unique(valid_values, sorted=True):
            group_mask = valid_mask & column.eq(group_value)
            if not group_mask.any():
                continue

            group_accs[int(group_value.item())] = float(correct[group_mask].float().mean().item())

        metrics_by_attribute[name] = group_accs

    return metrics_by_attribute


def _compute_group_performance_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    attributes: Mapping[str, torch.Tensor] | None,
    attribute_names: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, dict[str, float | dict[int, dict[str, float]]]]:
    """二値分類の属性 group ごとの性能と gap / worst-group 指標を内部的に計算する。

    各属性値の `n` と陽性率は常に返す。AUROC と BACC は group 内に陽性・陰性が
    両方ある場合だけ計算し、それ以外は NaN とする。集約 gap と worst-group は評価可能な
    group が 2 個以上ある場合だけ計算する。
    """
    if attributes is None:
        return {}
    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch_size, num_classes]")
    if logits.size(1) != 2:
        raise ValueError("group performance metrics currently require binary logits")

    targets = targets.view(-1).long()
    batch_size = targets.size(0)
    if logits.size(0) != batch_size:
        raise ValueError(f"logits batch size ({logits.size(0)}) does not match targets batch size ({batch_size})")
    values = attributes.get("categorical")
    if values is None:
        return {}

    values = _validate_attribute_tensor(values, batch_size, "categorical")
    missing = _resolve_missing_mask(values, attributes.get("categorical_missing"))
    names = _resolve_attribute_names("categorical", values.size(1), attribute_names)
    positive_scores = torch.softmax(logits, dim=1)[:, 1]
    predictions = torch.argmax(logits, dim=1)
    result: dict[str, dict[str, float | dict[int, dict[str, float]]]] = {}

    for index, name in enumerate(names):
        column = values[:, index]
        valid_mask = ~missing[:, index]
        group_values: dict[int, dict[str, float]] = {}
        evaluable_aurocs: list[float] = []
        evaluable_baccs: list[float] = []

        for group in torch.unique(column[valid_mask], sorted=True):
            group_mask = valid_mask & column.eq(group)
            group_targets = targets[group_mask]
            group_id = int(group.item())
            n = int(group_mask.sum().item())
            positive_rate = float(group_targets.float().mean().item())
            if torch.unique(group_targets).numel() < 2:
                group_values[group_id] = {"n": float(n), "positive_rate": positive_rate, "auroc": float("nan"), "bacc": float("nan")}
                continue

            auroc = float(binary_auroc(positive_scores[group_mask], group_targets).item())
            group_predictions = predictions[group_mask]
            tpr = group_predictions[group_targets.eq(1)].eq(1).float().mean()
            tnr = group_predictions[group_targets.eq(0)].eq(0).float().mean()
            bacc = float(((tpr + tnr) / 2).item())
            group_values[group_id] = {"n": float(n), "positive_rate": positive_rate, "auroc": auroc, "bacc": bacc}
            evaluable_aurocs.append(auroc)
            evaluable_baccs.append(bacc)

        if len(evaluable_aurocs) < 2:
            result[name] = {
                "groups": group_values,
                "auroc_gap": float("nan"),
                "bacc_gap": float("nan"),
                "worst_group_auroc": float("nan"),
                "worst_group_bacc": float("nan"),
            }
            continue
        result[name] = {
            "groups": group_values,
            "auroc_gap": max(evaluable_aurocs) - min(evaluable_aurocs),
            "bacc_gap": max(evaluable_baccs) - min(evaluable_baccs),
            "worst_group_auroc": min(evaluable_aurocs),
            "worst_group_bacc": min(evaluable_baccs),
        }
    return result
