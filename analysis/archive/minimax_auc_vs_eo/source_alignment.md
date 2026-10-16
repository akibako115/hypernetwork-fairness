# 原著・著者実装との対応

照合日: 2026-10-06。実装先は `feat/minimax-auc-eo-age`。
年齢はraw値で65歳未満／以上に分ける。陽性はNo Finding、スコアは `z1-z0`。

## Minimax AUC

[Yang et al., AAAI 2023](https://krvarshney.github.io/pubs/YangKVY_aaai2023.pdf)のAlgorithm 1と、
[著者repoのRWMAUCLearner](https://github.com/zhenhuan-yang/MinimaxFairAUC/blob/main/src/rwm_models.py)を参照した。

| 項目 | 今回の実装 |
|---|---|
| 条件付きリスク | 各陽性群g・陰性群hについて `mean(softplus(-(s_pos-s_neg)))` |
| q初期値 | trainの `P(g\|Y=1) P(h\|Y=0)` |
| モデル更新 | 旧qと同じforwardのリスクによるgradient |
| 双対更新 | モデル更新後、`log_q <- log_softmax(log_q + eta*R_old)` |
| ペア順序 | `(positive g, negative h)`。著者コードの `(negative i, positive j)` と転置関係 |
| 標本抽出 | 4セル各32件をIID復元抽出。4ペア、各1024標本ペア |

著者コードは更新幅を `temp * model_lr` とし、`freq` ごとにリスクを平均する。
今回は論文の独立した双対ステップ幅を直接設定し、毎step更新する。
共有ResNetは各画像を一度forwardし、ペア間でスコアを再利用する。著者コードのペアごとの画像反復と異なり、
BatchNormへの寄与が変わる。optimizerは共通比較のAdamW、初期値は既存ERM、出力はvalidationで選ぶ単一epoch。
これらは原著の実験設定の完全再現ではない。q初期値には母集団の群・クラス件数が残るため、陽性率から完全に独立とは呼ばない。

## EO rate制約

[Cotter et al., JMLR 2019](https://jmlr.org/papers/volume20/18-616/18-616.pdf)のAlgorithm 3と、
[公式TFCOのproxy_lagrangian_optimizer.py](https://github.com/google-research/tensorflow_constrained_optimization/blob/master/tensorflow_constrained_optimization/python/train/proxy_lagrangian_optimizer.py)を照合した。
特に制約側の勾配は公式の `tensordot([0, constraints], distribution, axes=0)` と同じ向きの外積。

| 項目 | 今回の実装 |
|---|---|
| hard判定 | `s>0`。tieは陰性 |
| 制約順序 | 陽性の0→1、1→0、陰性の0→1、1→0 |
| 制約 | `r[g,c]-r[h,c]-epsilon_c <= 0`。年齢2群で4本 |
| モデル側 | `lambda_task*corrected_CE + lambda_constraints·proxy_g` |
| 制約側 | hard違反だけを使う。taskの成分は0 |
| swap-regret状態 | 列確率行列M、`M lambda=lambda` の定常分布 |
| 更新 | `logM += eta * outer([0,hard_g], lambda_old)`、列方向log-softmax |
| 数値計算 | Mとpower iterationはfloat64、残差L1≤1e-8、上限1000回。未収束で停止 |

Mの初期値は論文に沿う一様分布。TFCOの既定はtask寄りの初期値と最小radiusのfloorを使うが、今回は採用しない。

代理制約は、2026-10-13 から TFCO の既定 `HingeLoss`（margin 1）と同じ hinge 上下界を使う（`proxy: hinge`）。
`U(s)=(1+s/tau)_+`、`D(s)=1-(1-s/tau)_+`。D は、負の係数がかかる rate を TFCO が補集合の hinge で評価する形と同じ。
全スコアで `D(s)≤I(s>0)≤U(s)` なので、`mean(U_g)-mean(D_h)-epsilon` はhard制約の上界。
U は閾値の正側でも勾配を持つので、率の高い群の標本を閾値の下へ押し戻せる。

それ以前の run は、自前で設計した smoothstep 上下界 `U(s)=H(1+s/tau)`、`D(s)=H(s/tau)`（`H(x)=3x²-2x³` を[0,1]でclip）を使っていた。
U は s≥0 で 1、D は s≤0 で 0 に張り付くため、代理制約は予測を閾値の反対側へ動かす勾配を持たない。
代理制約だけを score について下げても hard rate は1件も変わらないことを、テストで確認した
（`test_smoothstep_proxy_cannot_cross_the_threshold`）。したがって、旧 run の EO 条件で群間差が縮んだ／縮まなかった結果を、
EO 制約の効果として読むことはできない。旧 run を再現するときは `model.loss_fn.proxy=smoothstep` を指定する。
ResNet・AdamW・非凸proxy・有限epoch候補のvalidation選択に対し、原著の収束・実行可能性保証を主張しない。

## FairDRO

[Jung et al., ICLR 2023](https://arxiv.org/abs/2303.00442)のAlgorithm 1、式(8)–(10)を照合した（2026-10-07）。

| 項目 | 今回の実装 |
|---|---|
| 群重み | クラスごとに年齢群の `q[y][g]`。初期値は一様 1/G |
| モデル側 | `(1/2) Σ_y Σ_g q[y][g]·CE(g,y)`。CEはセル平均、class weightなし |
| 抽出 | 4セル各32件。Alg.1 l.6の「各(y,a)から等数」に対応 |
| 内側最大化 | 閉形式（Prop.2）`q* = 1/G + sqrt(rho/G)·dev/‖dev‖₂`。偏差0なら一様 |
| 重み更新 | 毎epoch末に `q <- (1-eta_t)q + eta_t q*`、`eta_t = 1-t/T`、T=30 |
| 更新に使う損失 | trainの全件を評価transform・eval modeで推論したhard誤り率（`z1-z0>0`） |

原著の重みは負になり得て、EGではなく平滑化したIBRで更新する。この点を保つため、simplexに射影しない。
q* の最小成分は `(1-sqrt(rho(G-1)))/G` なので、`rho ≥ 1/(G-1)` では誤り率のわずかな差でも重みが 0 以下になる
（G=2・rho=1 では各クラスで片方の群が 0 になり、一方の群だけを学ぶ解へ振れた）。
2026-10-13 から、この範囲の rho を指定すると起動時に警告し、毎 epoch の `q_min` を記録する。禁止はしない。

DataModule が held-out split を持つとき、q は held-out 全件の hard 誤り率で更新する（`fair_dro_q.jsonl` の `source`）。
train 全件では少数セルの記憶で誤りが消え、q が戻ったため（conditional_fairness_one_stage）。これは原著からの変更で、
EO・Minimax AUC の held-out と同じ扱い。held-out がなければ原著どおり train 全件を使う。
原著からの変更点：
- ERMからの継続学習
- AdamW
- validation選択（最終epochも併記）
- trainの誤り集計にaugmentationを使わない

原著は「全trainで群の損失を計算してqを更新」とだけ書き、train/eval modeを指定しない。
誤り集計はvalidation後のepoch末に行うので、epoch tの候補checkpointには、そのepochの学習に使ったqが残る。
推移は `metrics/fair_dro_q.jsonl` に、epoch・eta・q*・q・セル誤り率・件数として記録する。

## 抽出補正・評価

trainのセル件数だけから `P_gc` を記録する。抽出分布は `Q_gc=1/4`。
CEは `mean((P/Q)*w_y*CE)/Z`、`Z=sum(P_gc*w_c)`。群CEは母集団の `P(c|g)` を復元する。
validationも同じtrain由来class weight・固定Zを使う。群のTPR/FPRにはclass weightを掛けない。

fit前のsanity validationを全件実行し、読み込んだ親ERMをepoch 0として記録する。
その後の全epoch checkpoint、SHA-256、自然分布validation指標を保存し、testを読まずに選ぶ。
候補checkpointは重みと双対bufferを保存し、optimizer状態を省く。lastはoptimizer状態を含む。
途中再開は候補履歴・親評価を復元できないため拒否する。親の同seed・ERM確認は実験名ではなく
`continuation.require_parent` に基づき、親configが存在しない場合も起動を拒否する。
EOのvalidation LP mixtureと平均確率スコアensembleは別々に評価する。
論文の全step混合とは異なる30epochの有限候補に対する実用的選択である。

原著URLとrepoのbranch URLは照合時の参照。将来の更新で内容が変わる可能性がある。
コードをコピーせず、数式と更新順序を独立実装した。
