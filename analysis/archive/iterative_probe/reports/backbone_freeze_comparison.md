# 2-stage adaptation / 1-stage E2E comparison

## 条件

- 対象: `fc` / `stage4+fc` × GroupDRO step size `1e-3` / `1e-2` の各条件について、ImageNet初期化から
  backboneを更新する1-stage E2Eと、ERM ResNet checkpointからbase backbone・共有classifierを固定する2-stage
  adaptationを比較する。各条件は seed 42 の1本だけである。
- 学習: warmup 2 epoch、stage01 / stage02 を各5 epoch、cohort 10、`weighting=inverse`。
- checkpoint: 各runの stage02 で val AUROC が最大の checkpoint。hidden cohort の worst/gap は選択基準ではなく
  評価指標として読む。
- validation: global、`q`、hidden cohort、属性群の推移。
- test: val 全体の感度 0.9 で決めた単一閾値を使い、age × sex × race の単独群・交差群を評価する。

## 観察事実

run 完了後、`backbone_freeze_comparison.ipynb` を kernel restart から実行して記入する。群別の数値には
`n` を添える。単一seedなので、条件間の差を再現性のある効果とは扱わない。

## 解釈候補

未記入。

## 追加確認

単一seedの探索で候補が出た場合だけ、2-stage adaptation / 1-stage E2E の同じ条件対を追加seedで再現確認する。
