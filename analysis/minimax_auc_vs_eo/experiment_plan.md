# 順位改善とEO制約の達成を別々に判定する

2026-10-06。Minimax AUCとEO rate制約の実験計画。
実装は[実装計画](implementation_plan.md)に従う。新規presetとCLI設定は実装予定で、以下のコマンドは現時点では起動できない。
この文書の許容差・探索範囲は今回の設計値であり、原著の推奨値や臨床的に承認された基準ではない。
学習runは未起動、GPU・所要時間は未確認である。
初回は年齢のみの2群（65歳未満／65歳以上）を対象とする。
年齢2群の比較結果を確認してから、年齢×raceのintersectionへ進む。

## 検証仮説と結論の範囲

H1：群CEの混合を使わず、クラス条件付きの順位riskを最悪ケースの対象にすると、最小群ペアAUCを改善できる。
主にMinimax AUCと固定重みAUCMaxを比較し、ERM・DROも参照する。

H2：CEの大きさでなくhard rateの制約違反に圧力を配分すると、共通閾値でEOを達成できる。
EOと同じsampling・task正規化のERMを直接比較し、群別閾値による参照も置く。

H3：改善がsamplingや追加学習予算だけでは説明できない。
層化ERM/DROと自然samplingの継続ERM/DROを比較する。
この実験だけで、既存DROの不調の原因が陽性率に確定したとは結論しない。

## 共通条件と初期化

| 項目 | 主比較の設定 |
| --- | --- |
| project / study | `hypernet_e2e` / `minimax_auc_vs_eo` |
| タスク | CheXpert No Finding二値分類。陽性＝所見なし |
| split | `data/chexpert/splits_race_known`。train 157,852件、val 20,058件 |
| 群 | 年齢のみ：65歳未満／65歳以上の2群。群IDは `age_group_65` |
| model | ResNet-50、metadata入力なし、全層を更新 |
| 初期値 | 同seedの既存ResNet ERM・best val AUROC checkpoint |
| 追加予算 | 各条件30 epoch、optimizer stateは引き継がずAdamWを新規作成 |
| optimizer | lr 1e-4、weight decay 1e-2、schedulerなし |
| batch / precision | 128 / bf16。objective集約はfloat32、EO定常分布はfloat64 |
| 実行方式 | 単一GPU、gradient accumulation=1 |
| seed | 42で探索し、同じ設定を43・44へ展開 |
| 評価閾値 | スコア `z1-z0>0`。閾値の変更は別条件 |
| logger | CSVとW&B、project `fairness_hypernet` |

親runは既存M01の同seed runを選び、train/valのhashとbest checkpointの実体hashを照合した。
親の具体的run-idとW&B URLは[runs.md](runs.md)を参照する。checkpoint SHA-256の正本は親run.json。
既存M02は過去比較の参照に使えるが、追加30 epochのDRO対照の代わりにはしない。
現行コードにはvalidationでGroup DROのqを更新する経路があるため、これを修正した版でC2/C4を実行する。
過去M02の結果との差を、新手法の効果だけで説明しない。
これはERMからの継続学習の比較で、ImageNetから30 epochの既存比較とは予算が異なる。
Minimax原著のAUCMax初期化とは異なる。必要なら後述の追加実験で分ける。
学習の群定義からraceを外しても、splitは `splits_race_known` のままとする。
今回の結論はrace既知のcohortにおける年齢差について述べる。
C2/C4/C5も年齢2群で学習し、過去の交差8群DROと群定義を混ぜない。

### 少数セルと復元抽出を監視する

以下は2026-10-06にtrain/val CSVから確認した件数。testの追加集計はこの計画の作成では行っていない。

| 群ID | 群 | train陰性 | train陽性 | val陰性 | val陽性 |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | 65歳未満 | 75,145 | 12,385 | 9,566 | 1,442 |
| 1 | 65歳以上 | 66,453 | 3,869 | 8,512 | 538 |

層化条件は1セル32件、1 epoch 1,234 step、計157,952抽出とする。
各セルを39,488回抽出するため、65歳以上の陽性3,869画像は平均約10.2回/epoch使われる。
これを独立な情報量の増加とはみなさない。セル別train/valの乖離、ユニーク画像数と重複率を監視する。
強い過学習が出た場合は復元抽出の配分を変える新計画を作り、主比較の途中でsamplerを替えない。

## 条件表

| ID | 条件 | sampling | taskの規約 | 役割 |
| --- | --- | --- | --- | --- |
| C0 | 親ERMのまま | 学習なし | 親run | 追加学習前の参照 |
| C1 | ERM継続 | 現行uniform | 現行global weighted CE | 追加予算の対照 |
| C2 | Group DRO継続 | 現行uniform | 現行global weighted群CE、step 0.01 | 追加予算のDRO対照 |
| C3 | 補正ERM | 年齢×クラスの4セル層化 | 自然分布へのimportance補正、固定Z | 新手法の直接のCE対照 |
| C4 | 補正Group DRO | 年齢×クラスの4セル層化 | 元の群内クラス混合を復元、step 0.01 | samplingをそろえたDRO対照 |
| C5 | 補正群等重み | 年齢×クラスの4セル層化 | C4と同じ群CEを等重み平均 | 適応重みの必要性 |
| C6 | AUCMax | 年齢×クラスの4セル層化 | population群ペア重み固定、CEなし | AUC目的への変更の対照 |
| C7 | Minimax AUC | 年齢×クラスの4セル層化 | 4群ペア、q適応更新、CEなし | H1の主対象 |
| C8 | EO rate制約 | 年齢×クラスの4セル層化 | C3のCE＋proxy-Lagrangian、4制約 | H2の主対象 |
| C9 | FairDRO | 年齢×クラスの4セル層化 | クラスごとの年齢群重み×セルCE。重みは全trainの0-1誤り率から毎epoch更新 | H2の先行手法 |

C6/C7/C9にはclass weightを掛けず、configは `weighting=none` とする。
C9は4セル均等抽出のセル平均CEを直接使い、importance補正も入れない。
それ以外はglobal inverse class weight。公平性の条件付きrateにはclass weightを掛けない。
親checkpointとoptimizer設定、追加step数は全条件でそろえる。
層化条件どうしではsamplerのseed規約も共通にする。
sampling補正でBatchNormや勾配分散までは等しくならないため、C1/C3とC2/C4の差を併記する。

## 実験を段階的に増やす

### 0. 実装検証と短い動作確認

まず固定データのテスト・preflight・全presetのdry-runを通す。
実画像ではC3/C7/C8をseed 42・各1 epochで確認する。通常のCSV/W&B付きrunとして記録する。
この3本は指標の優劣や最終候補の選定に使わず、更新回数・空セル・非有限値・checkpoint・計算時間を確認する。
最初のvalidationを通過し、1 epochの実測時間・ピークGPUメモリ・disk使用量を得てから本実験の時間を見積もる。

### 1. seed 42で対照と探索を行う

C1〜C6を各1本、C7を3設定、C8を9設定、すべて追加30 epochで回す。

| 手法 | 初回の探索 | 本数 |
| --- | --- | ---: |
| C1〜C6 | 各条件の既定値 | 6 |
| C7 | `dual_step_size={0.001,0.01,0.1}` | 3 |
| C8 | `epsilon_pos=epsilon_neg=0.10`、`dual_step_size={0.001,0.01,0.1}` × `tau={0.5,1.0,2.0}` | 9 |

探索は18本。C8の許容差を最初から同時に振って探索量を増やさない。
EOの主解析はepsilon 0.10と固定し、その達成可否を報告する。
epsilon 0.05/0.20は選んだ更新率・tauを固定した感度分析の追加2本とする。
epsilon 0.20だけ成立した場合、epsilon 0.10を達成したとは扱わない。

2026-10-07追加：C9もseed 42の親ERMから追加30 epochで回す。

| 手法 | 初回の探索 | 本数 |
| --- | --- | ---: |
| C9 | `radius={0.01,0.1,0.3,1,3}` | 5 |

C9の範囲は原著Tab. C.1の `[1e-2, 1e2]` から選んだ。年齢2群では重みが `(1±√rho)/2` の2値をとる。
`rho=1` で `[0,1]`、`rho=3` で誤りの少ない群に負の重み（約-0.37）が付く。
C9は30 epochの全trainの誤り集計を学習中に行うため、1 epochの時間を動作確認で実測する。
選んだrhoをseed 43・44へ展開し、追加2本とする。C8との比較は同じEO選択規約で行う。

途中で非有限値、定常分布の非収束、欠損セル、更新回数不一致が出たらrunをfailedとして残す。
目的指標が悪いだけで都合よく停止せず、30 epochの予算を守る。
実装不具合の修正後は新runを作り、修正前後を同じ条件の反復として混ぜない。

### 2. 選んだ設定を3seedで確認する

seed 42のvalidationでC7/C8それぞれ1設定を選び、seed 43・44へ同じ設定を展開する。
C1〜C6も43・44で回すので、追加16本。seed 42の選択済みrunは再利用する。
主比較は8条件×3seed＝24本、探索全体では34本となる。
動作確認3本を加えると37本。epsilon感度分析2本はこの数に含めない。
失敗runや追加ablationは別枠で明記する。

新しいseedのvalidationでも、固定したcheckpoint選択規約を使う。
seedごとに更新率・tau・許容差を調整しない。
W&B groupは `maeo_age_c7_eta001`、`maeo_age_c8_eps010_eta001_tau1` のように年齢2群と設定を区別し、seedを含めない。
将来のintersection条件には別のgroup名を付け、年齢のみのrunと混ぜない。

## validationでの選択規約

各runのepoch 0（親参照）とepoch 1〜30を候補にする。
主結果のcheckpointは以下で選び、従来のbest val AUROCとlastも別に報告する。
許容低下はAUROCの絶対差0.005、rateの絶対差0.02とする。
これらは探索前の設計値であり、統計的な有意差判定の閾値とは区別する。

### Minimax AUC

候補は親ERM比で全体AUROCと群内worst AUROCの低下がそれぞれ0.005以内のepoch。
この集合から4群ペア最小AUCを最大化し、同点なら群内worst AUROC、全体AUROC、早いepochの順に選ぶ。
run間の更新率選択にも同じ規約を使う。
親と同じepoch 0が選ばれた場合は改善なしとする。
代理risk最大値は診断指標として残すが、実測最小AUCをproxyで置き換えない。

採用の比較はC7対C6を第一、C7対C3/C4を第二とする。
4群ペア最小AUCだけが上がり年齢2群のworst AUROCが下がる場合は、そのトレードオフを記す。
主仮説に沿う改善と、現行の群内worst AUROCの改善を同じものと呼ばない。

### EOの単一checkpoint

全validation画像を集計したhard rateで次を満たすepochを候補とする。

- TPR gap、FPR gapともに0.10以下。
- 全体AUROCと群内worst AUROCが、親ERM比で0.005を超えて下がらない。
- 各群のTPR・TNRが、親ERM比で0.02を超えて下がらない。

候補から固定Zの自然分布weighted CEを最小化し、同点なら全体AUROC、早いepochの順に選ぶ。
該当候補がなければ「単一checkpointでは制約と性能下限の同時達成なし」と報告する。
補助報告として、AUROC下限内の最小hard違反のepochを保存するが、成立したモデルとは呼ばない。
同点判定の数値許容は1e-8、制約判定の丸め許容は1e-6とし、samplingの統計誤差を許容差へ足さない。

TPRとTNRの下限は、差を消すために他群を下げる挙動を検出するための採用条件である。
これが厳しく候補がなくても、下限を後から緩めて主結果を成立させない。
制約だけ成立・性能下限だけ成立・両方成立を分け、全候補のPareto曲線を示す。

C9も同じ規約で単一checkpointを選ぶ。FairDRO原著は最終epochを報告するので、C9の最終epochも併記する。

### EOの確率的混合は別の結果として選ぶ

まず親比の全体・群内worst AUROC下限を満たすepochに候補を絞る。
候補のhard TPR/FPRをvalidationで計算し、非負で和1の混合重みpiを求める。
目的は候補CEの加重平均の最小化、制約は年齢2群の4本のrate差と各群TPR/TNRの下限とする。
このLPはstudy内の分析処理で実装し、依存には既存環境のSciPyを明示する。
LPがinfeasibleなら未達として残す。solver toleranceと実際の最大違反も記録する。

各候補のhard判定を重み付き平均した確率が、ランダム化分類器の陽性判定確率になる。
平均logitや平均スコアに閾値を掛ける方式は別の分類器なので使わない。
testのrateは期待値で厳密に集計でき、モデルをランダム抽出して測定雑音を加える必要はない。
患者単位bootstrapでも混合重みはvalidationで決めた値に固定する。

AUROCはこのhard分類器の選択目的と混ぜない。
参考として平均確率スコアのAUROCを出す場合は「スコアensemble」と別名にし、
各候補のAUCの平均をensemble AUCと呼ばない。全体AUC下限は各構成候補とスコアensembleで別に確認する。
候補は30epochの離散集合であり、全stepを使う原著の理論的混合の再現ではない。
trainでのlambda_task平均による混合は診断用に残し、validation LPとの違いを示す。

## 評価と成果物

testはハイパーパラメータ、選択規約、参照閾値、混合重みを固定した後に一度評価する。
3seedそれぞれの値と平均を報告し、seed間標準偏差だけで標本の不確実性を代用しない。
同じ患者を同じbootstrap標本へ抽出し、比較条件間で対応を保った2,000回のpaired bootstrapを行う。
患者IDは画像pathから既存の抽出規約を検証して復元し、画像単位のbootstrapへ黙って切り替えない。
比較差の95%区間と最悪群の入れ替わりを示す。
3seedの平均指標差を主対象とし、seedの予測スコアを平均するensembleは別解析にする。

| 成果物 | 内容 |
| --- | --- |
| `01_validation_selection.ipynb` | epoch曲線、下限、制約実行可能性、選択理由、EO混合LP |
| `02_test_comparison.ipynb` | 全体・年齢2群・4群ペア指標、paired差、年齢別指標 |
| `03_operating_points.ipynb` | 共通閾値・群別Youden・EO後処理参照との比較 |
| `results/validation_candidates.csv` | 候補epoch、全指標、制約・下限判定 |
| `results/selection.json` | 手法ごとの設定、checkpoint hash、混合重み、参照run |
| `results/test_metrics.csv` | seed・条件・群・群ペアごとの数値と件数 |
| `results/paired_differences.csv` | 比較差、95%区間、bootstrap設定 |
| `reports/comparison.md` | 仮説ごとの判断、性能の代償、未達・限界 |

後処理の閾値はvalidationだけで決める。
群別YoudenとTPR合わせに加え、厳密EOのランダム化後処理を参照にする場合は
[Hardt et al.](https://arxiv.org/abs/1610.02413)に沿って別実装・別ラベルで扱う。
単一群別閾値だけでTPR/FPR双方がそろうと前提にしない。

レポート種別の層別評価は、[機構診断](../fairness_mechanism_probe/reports/mechanism_findings.md)を踏まえた補助解析にする。
主タスクのラベルを探索中に差し替えず、補助部分集合で良かった結果を主結果の代わりにしない。

## 主比較の後に行う追加実験

最初の展開先は年齢×raceの8群とする。ただし、年齢2群の3seed比較で、
更新が安定し、性能下限を守りながら目的指標が改善する候補があるかを先に判断する。
MinimaxはC6に対する最小群ペアAUCの改善、EOはC3に対するhard gapと制約達成を確認し、
paired区間を含めて結果を示す。改善が不確かなら結論を保留し、展開を自動実行しない。

intersectionへ進む場合は、新しい群定義・セル件数・sampling・探索範囲を先に定める。
Minimaxは8×8の64群ペアへ拡張する。EOは全体参照・全群ペア・max-min差のどの表現を使うか再検討し、
以前の112制約案を既定として引き継がない。
年齢2群の「worst」と交差8群の「worst」は別指標として報告し、直接同一視しない。

優先順は、Minimaxの群内のみ・q初期値一様、EOのepsilon感度、AUCMax初期化からの継続、
最後にEO marginと有望手法のSpatial LoRA展開とする。
AUCMax親を使う場合は、他条件にも同じ親と同じ追加予算を与える。
どの追加も新しいpreset／設定として記録し、主比較へ後から混ぜない。
主比較で改善がなければ、全アーキテクチャへの展開は行わず、未達の理由を分析する。

## 起動までにそろえるもの

実装完了後、各条件のdry-run表、preflight結果、親checkpoint照合、GPU空きと既存プロセス、
動作確認で測った時間・メモリ・disk、生成run数を提示する。
毎epoch保存は最大30候補を持つため、optimizer込みcheckpointの実測サイズで必要diskを見積もる。
GPU番号と実行先はこの段階で決め、現在の記録を過去の割り当てから流用しない。

主比較の起動形は次のとおり。新規名は実装時に確定し、dry-runで確認する。

```bash
uv run python -m projects.hypernet_e2e.run \
  experiment=resnet_chexpert_from_resnet_age_minimax_auc_race_known \
  study=minimax_auc_vs_eo experiment_seed=42 \
  model.backbone_checkpoint_path=<同seedの親ERM-checkpoint> \
  model.loss_fn.dual_step_size=0.01 \
  logger.wandb.group=maeo_age_c7_eta001 dry_run=true

uv run python -m projects.hypernet_e2e.run \
  experiment=resnet_chexpert_from_resnet_age_equalized_odds_rate_race_known \
  study=minimax_auc_vs_eo experiment_seed=42 \
  model.backbone_checkpoint_path=<同seedの親ERM-checkpoint> \
  model.loss_fn.epsilon_pos=0.10 model.loss_fn.epsilon_neg=0.10 \
  model.loss_fn.dual_step_size=0.01 model.loss_fn.tau=1.0 \
  logger.wandb.group=maeo_age_c8_eps010_eta001_tau1 dry_run=true
```

この計画の合意と、具体的な学習ジョブの起動確認は別に扱う。
repoの[train skill](../../.agents/skills/train/SKILL.md)が
「起動コマンド全文と、`dry_run=true` が出した表をそのまま提示し、明示承認を取る。」と定めているため、
実装・テスト・dry-runまで準備してから具体的な起動条件を確認する。
