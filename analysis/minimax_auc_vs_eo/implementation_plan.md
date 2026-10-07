# Minimax AUCとEO rate制約を既存の学習契約へ組み込む

2026-10-06。実装前の設計。主比較はMinimax AUCとEO rate制約、EO marginは追加候補とする。
数式・更新順序・sampling・checkpoint選択を先に固定し、実装後は[実験計画](experiment_plan.md)の順で検証する。
この文書に挙げる新規module・preset・CLI引数はまだ存在しない。
初回は年齢のみの2群（65歳未満／65歳以上）。年齢×raceのintersectionへの展開は結果を確認した後に行う。

## 原著の機構を維持し、今回の変更を記録する

| 項目 | 採用する機構 | 今回の変更 |
| --- | --- | --- |
| Minimax AUC | 群内・群間のpairwise logistic risk、群ペア重みのmirror ascent | 年齢2群の4群ペア。ResNet・AdamW・ERM checkpoint初期化を使う |
| EO | proxy-Lagrangianのモデル側／制約側の分離、swap-regret更新 | 年齢2群の両方向rate差、独自の上界proxy、AdamWによるモデル更新 |
| モデル選択 | validationによる候補比較 | 単一checkpointを主報告、EOの確率的混合を別報告 |

Minimax AUCは[Yang et al., Algorithm 1](https://krvarshney.github.io/pubs/YangKVY_aaai2023.pdf)と
[著者実装](https://github.com/zhenhuan-yang/MinimaxFairAUC)を参照する。
EOは[Cotter et al., §4](https://jmlr.org/papers/volume20/18-616/18-616.pdf)と
[公式proxy-Lagrangian実装](https://github.com/google-research/tensorflow_constrained_optimization/blob/master/tensorflow_constrained_optimization/python/train/proxy_lagrangian_optimizer.py)を参照する。
原著の実験全体の再現とは呼ばず、「原著の機構に沿ったCheXpertへの適用」と表記する。
外部コードを依存として追加せず、数式と小さい固定データの結果を照合する。

## 共通の入力と分布を明示する

スコアは `s=z1-z0`、陽性は `target=1`（No Finding）、群IDは `age_group_65`（0：65歳未満、1：65歳以上）。
年齢群は標準化前のraw ageで生成する。modelにはmetadataを入力しない。
hard予測は `s>0` とし、`s=0` は現行argmaxと同じ陰性にする。
AUCの同点は0.5として評価する。初回実装は単一GPU・gradient accumulation=1に限定する。

`ObjectiveInput`の現行shapeとbatchの3要素構成は維持する。
sampling補正は、各objectiveがtrainの群×クラス件数から計算するbufferを使う。
件数はvalidation/testから解決しない。train内に空セルがあれば起動前に失敗させる。

### 年齢×クラスの4セルを確保するsampler

新規 `data/group_class_sampler.py` に `GroupClassBatchSampler` を置く。
batch 128のうち各セルから32件ずつ、セル内では独立に復元抽出する。
各batchの行順はshuffleし、epoch開始時のseedを `seed+epoch` から決める。
1 epochは `ceil(N_train/128)` batchで、最終batchも128件にそろえる。
samplerの抽出分布は `Q(g,c)=1/4`。validation/testは現在の固定順・自然分布を使う。

`GroupClassImageDataModule` を新規に設け、既存のgroup ID生成とDatasetを再利用する。
既存 `ImageDataModule` のsampling既定を変えない。`data`の専用presetでclass・samplerを差し替える。
samplerはtrain Datasetのindexだけを返し、画像・augmentationは従来のDatasetが所有する。
各セルの件数、抽出件数、ユニーク画像数、重複率を記録する。

### 分類損失の補正

trainの実測分布を `P(g,c)=n_gc/N`、global class weightを `w_c` とし、
`rho_gc=P(g,c)/Q(g,c)=4*n_gc/N`、`Z=sum_gc P(g,c)*w_c` を固定値として保存する。
層化samplingのERMとEOの分類目的は次とする。

\[
L_{\mathrm{task}}=\frac{1}{BZ}\sum_{i=1}^{B}\rho_{g_i,y_i}w_{y_i}\,\mathrm{CE}_i.
\]

この推定は固定したサンプル単位の損失に対して、自然分布での正規化weighted CEを対象にする。
現在の `F.cross_entropy(weight=..., reduction='mean')` はbatch内の重み和で割るため、有限batchで厳密には同じにならない。
現行のuniform ERMを残し、層化ERMを公平性手法の直接の対照にする。
BatchNormの統計・抽出の重複・勾配分散への影響はimportance補正では消えない。

層化DROと群等重みでは、各群内の元のクラス混合を復元する。

\[
L_g=\sum_{c\in\{0,1\}}P(c\mid g)w_c\,\overline{\mathrm{CE}}_{g,c}.
\]

この `L_g` を既存と同じ群重み更新または等重み平均へ渡す。
単純に層化batch内のCEを群平均するとクラス比が1:1になるため、元のDROとの比較には使わない。
新規の補正版objectiveに閉じ、既存 `objectives/group.py` の計算は変えない。

## Minimax AUCの実装

新規 `objectives/minimax_auc.py` に `MinimaxAUCTaskLoss` と固定重みの `PopulationAUCTaskLoss` を置く。
公開設定は `num_groups, group_key, population_cell_counts, dual_step_size, pair_scope`。
主比較の `pair_scope=all`、CE追加なし。群ペア `(g,h)` は陽性群がg、陰性群がhで、flatten順は `num_groups*g+h`。
初回は `num_groups=2`、群内2ペア・群間2ペアの計4群ペアになる。

\[
R_{g,h}=\frac{1}{n^+_g n^-_h}
\sum_{i\in(g,1),j\in(h,0)}\operatorname{softplus}(-(s_i-s_j)),\quad
L_\theta=\sum_{g,h}q_{g,h}R_{g,h}.
\]

初期値は `q0[g,h]=(n_g1/N_pos)*(n_h0/N_neg)`。
これは全体AUCの群ペア構成比であり、適応更新前の重みには群頻度が入る。
今回避けるのはCEの陽性率混合であり、「全工程が基底率と無関係」とは説明しない。
`PopulationAUCTaskLoss` は同じq0を更新せず、全体AUC最適化の対照にする。

モデルの勾配には更新前のqを使い、同じforwardのdetach済みRで次stepのqを更新する。

\[
\log q_{t+1}=\operatorname{logsoftmax}(\log q_t+\eta_q\operatorname{stopgrad}(R_t)).
\]

log qをpersistent bufferとして保存する。順位差・softplus・集約はfloat32で実行する。
32件×32件×4群ペアで1batchあたり4,096サンプルペア。群ペア数とサンプルペア数を区別する。
全ペア行列を無制限に作らず、群ペア単位で計算する。
validationの代理riskはchunk処理し、全ペアの厳密な平均を得る。
AUROCはスコアのsortと累積カウントで計算し、全validationペアをGPUへ展開しない。

`pair_scope=intra` は追加ablation用に実装可能にするが、主比較に含めない。
順位損失はスコアへの共通加算に不変なので、切片・較正・固定閾値のrate改善を保証しない。

## EO rate制約の実装

新規 `objectives/equalized_odds.py` に `EqualizedOddsRateTaskLoss` を置く。
主設定は `epsilon_pos=epsilon_neg=0.10`、`threshold=0`。
年齢2群の両方向 `(0,1), (1,0)` に制約を置く。TPR 2本、FPR 2本、計4本とする。
これは2群のTPR差・FPR差の絶対値を、それぞれ許容差以下にする条件である。
原著の群対全体の比較と今回の群対群の比較は区別する。ここでは群間gapを直接制約する。

\[
r_{g,c}=\mathbb E[\mathbf1(s>0)\mid G=g,Y=c],\qquad
g_{c,g,h}=r_{g,c}-r_{h,c}-\epsilon_c\le0.
\]

全方向制約は各クラスでのmax-min gap制約と等価である。
順序は陽性側 `(0,1), (1,0)`、陰性側 `(0,1), (1,0)` とし、JSONに保存する。
条件付きrateにclass weightや群の出現頻度を掛けない。

### モデル側のproxyは上界として設計する

単純なsigmoid差はhard rate差の上界にならないため、主比較の既定にはしない。
今回の設計として、連続で一次微分可能なbounded rampを使う。
`H(x)=0 (x<=0), 3*x^2-2*x^3 (0<x<1), 1 (x>=1)` と定義し、
`U_tau(s)=H(1+s/tau)`、`D_tau(s)=H(s/tau)` とする。
任意のsで `D_tau(s)<=1(s>0)<=U_tau(s)` が成立する。

\[
\widetilde g_{c,g,h}=\overline{U_\tau(s)}_{g,c}
-\overline{D_\tau(s)}_{h,c}-\epsilon_c\ge g_{c,g,h}.
\]

このproxyは本projectの設計で、原著が特定した式の再現ではない。
境界近傍だけで勾配が生じるため、proxyとhard rateの差、勾配が生じるサンプル比率を記録する。
tauの初期値は1.0、探索候補は0.5・1.0・2.0。
上界が保守的すぎる場合や勾配が消える場合は、既定を黙ってsigmoidへ替えず、別条件として比較する。

### 制約側はhard rateによるswap-regret更新

`lambda` は分類目的の1成分と4制約の計5成分。
列確率行列 `M[5,5]` を一様に初期化し、`M*lambda=lambda` の定常分布を各stepで求める。
モデル側の損失は `lambda[0]*L_task + sum_k lambda[k+1]*proxy_g[k]`。
制約側のpayoffは `v=[0, hard_g_1,...,hard_g_4]` とする。

\[
\log M'_{ij}=\log M_{ij}+\eta_\lambda v_i\lambda_j,
\qquad \log M_{t+1,:,j}=\operatorname{logsoftmax}(\log M'_{:,j}).
\]

Mの列・行を取り違えない。定常分布はfloat64のpower iterationで求め、
`||M*lambda-lambda||_1<=1e-8` を収束条件、最大1,000 iterationとする。
非収束・非有限値は失敗扱いとし、前回値で継続しない。
主比較ではEMA・weight clipping・追加のCE係数を入れない。
log Mと更新回数はpersistent buffer、診断値はdetach済みとする。

AdamW・有限epoch・非凸ネットワークと今回のproxyでは、原著のoracleやregret条件を満たすと主張しない。
上界を使うことと、実測validation/testで制約を満たすことを分けて報告する。
単一checkpointが不成立でも、学習失敗と確率的混合でのみ成立する場合を区別する。

## Lightning接続と評価時の状態更新を分離する

`module.py` の既存 `loss_components` は属性adversary専用の加算規約を持つ。
今回のobjectiveをそこへ無理に接続せず、`forward`でscalar損失を返し、
`batch_diagnostics()` と `commit_update()` を新規objectiveの小さい契約にする。

forwardは更新前の状態で損失とpending診断を作り、model optimizerのstepが成功した後にcommitする。
`LitModule.optimizer_step` で親の処理を呼んだ後、対応objectiveだけを1回commitする。
closureの再評価・validation・loggingからはcommitしない。
gradient accumulationや複数GPUは起動前に拒否し、対応を広げる場合は別変更とする。
既存Group DROの更新順序は変えない。

現在の `_objective_components(include_task=False)` はplain objectiveのforwardも呼ぶ。
既存Group DROのforwardはqを更新するため、validationでもqが更新される経路がある。
これはコードの読解で確認した挙動であり、過去の指標への影響の大きさは未検証である。
plain objectiveのvalidationではforwardを呼ばず、通常CEと評価callbackだけを実行するよう修正する。
Group DROのtrainでの計算・更新順は維持し、評価時の更新を止めたC2/C4を新しい対照として回す。
過去runは書き換えず、比較レポートにこの修正とGit差分を明記する。
属性adversaryの既存評価挙動は維持し、回帰テストで確認する。

診断callbackはforwardの保存値を読むだけにする。
Minimaxには `q_before/q_after, R, cell_count, pair_count, entropy, max_q`、
EOには `lambda, lambda_task, hard_g, proxy_g, rates, stationary_residual, active_gradient_fraction` を記録する。
rateのepoch集約は分子・分母を合計して求め、batchのgap平均をepoch gapと呼ばない。

## checkpointと分析入口を拡張する

既存のbest val AUROCとlastを維持し、毎epochの候補checkpointを追加保存する。
保存候補はepoch 1〜30。epoch 0は親ERMへの参照とし、重みを複製しない。
各epochの全validation指標・proxy値・EO用lambda_taskをcallbackで確定する。
候補のpath・SHA-256・epochはrun内の `metrics/fairness_candidates.json` に残す。

fit終了時にはtestを読まず、validationの規約に従って候補を選ぶ。
選択理由は `metrics/fairness_selection.json` に出力する。
候補checkpointは既存のrun recorderが記録できるcallbackを使い、独自serializerを増やさない。
run artifact契約に候補・選択JSONを追加する際は、doc・schemaテストも同時に更新する。

分析はpackageに取り込んだ完了runだけを使う。
`analysis/common/predictions.py` は現在best val AUROCを暗黙に読むため、
明示したcheckpointを選べる引数を追加する。既定の選択挙動は変えない。
cacheのキーにはcheckpoint SHA-256とsplit hashを含め、別epochの予測を取り違えない。
選択基準・LP・図は `analysis/minimax_auc_vs_eo/` に閉じる。

## 実装単位と完了条件

| 順序 | 作業 | 主な変更先 | 完了条件 |
| --- | --- | --- | --- |
| 0 | 評価時の状態更新を止める | `module.py`、Group DROの評価テスト | validation前後でqが一致し、属性adversaryの評価を維持 |
| 1 | 固定データで数式と更新を照合 | `tests/golden/`、projectのobjectiveテスト | 手計算の4群ペアrisk、4制約、1step更新に一致 |
| 2 | samplerとtrain統計の解決 | `data/group_class_sampler.py`、新DataModule、`training.py` | 全4セル、再現性、補正推定、空セル拒否を確認 |
| 3 | 層化ERM・DRO・群等重みの対照 | 新規 `objectives/sampling_corrected.py` | 自然分布の全標本集約と期待値が一致 |
| 4 | MinimaxとAUCMax | `objectives/minimax_auc.py`、facade | q初期値・更新順・共通切片不変性を確認 |
| 5 | EO | `objectives/equalized_odds.py` | proxy上界、hard更新、列正規化、定常分布を確認 |
| 6 | model stepとの接続 | `module.py`、診断callback | optimizer stepごとに1回更新、評価で状態不変 |
| 7 | config・dry-run・記録 | training strategy／data／experiment preset、run契約 | 全条件を合成し、件数・補正・許容差を表に表示 |
| 8 | validation候補と分析 | checkpoint callback、commonの明示checkpoint、study notebook | 選択規約、LP、不確実性評価を固定データで検証 |

新規presetは `resnet_chexpert_from_resnet_age_<method>_race_known` 系とする。
data presetは `group_attribute_names=[age_group_65]`、`group_cardinalities=[2]`、`num_groups=2` を明示する。
公平性の主評価も `age_group_65` にそろえ、既存8群のDRO診断ラベル・交差群checkpoint基準を引き継がない。
`from_resnet` でも `freeze_backbone=false` とし、全層を更新する。
既存Spatial LoRAの凍結契約を流用しない。checkpointが欠けたらランダム初期化で続行せず失敗する。
新規strategyは `sampling_corrected_erm, sampling_corrected_group_dro, sampling_corrected_uniform_group,
population_auc, minimax_auc, equalized_odds_rate`。
`loss.py` はfacadeのまま、新規実装を `objectives/` へ置く。

### 間違った結論を防ぐテスト

- ペアの陽性／陰性の向き、対角・非対角、同点0.5、全ペアAUCの分解を手計算に照合する。
- クラス内のサンプルを均等に複製して陽性率だけ変えると、条件付きrateとpair riskが変わらない。
  q初期値とtask分布が変わる部分は、意図した変化として別に確認する。
- EOの4方向の最大違反が年齢2群のmax-min gapの違反と一致する。score=0のtieも含める。
- U/Dの上下界とproxy_g>=hard_gを、境界・極端値・ランダム値で確認する。
- qとMの更新で、勾配は旧状態、次stepは新状態を使う。logitのgradにdual更新が混ざらない。
- Mの列和1、lambdaの総和1、既知の定常分布・残差・大きいstepでの安定性を確認する。
- validation、test、sanity check、診断callbackで状態が変わらない。state_dict往復で状態が一致する。
- 同seed同epochのsamplingは一致し、違うepochは変わる。各batchで4セル32件ずつになる。
- 混合モデルのrateはhard予測の重み付き平均。平均logitへ閾値を掛ける誤実装を検出する。
- checkpoint選択はtestを読まない。LP infeasible・欠損群・NaNでは候補を成立扱いにしない。
- 既存ERM・DRO・属性adversaryのpresetと評価を回帰確認する。

実装後に関連テスト、projectのpreflight、ruffを実行する。
ドキュメント作成段階では新規コードが無いため、これらを実施済みとは記録しない。
