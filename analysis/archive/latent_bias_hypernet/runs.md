# runs

この study の学習はすべて run 契約（projects/*/runs）の外で回した探索用の probe で、W&B には記録していない。
結論の根拠にする条件は、`hypernet_e2e` の run として回し直す必要がある。local GPU0（RTX A6000）、2026-10-08。

## 1段目（`probes/train_stage1.py`、`cache/stage1/<name>/`）

ImageNet 初期化の ResNet-50、batch 128、AdamW（lr 1e-4、wd 1e-2）、bf16、fit split の陽性率から作る global inverse
class weight、CXR_Fairness と同じ augmentation。val AUROC が最大の epoch の重みで全 split の表現と logit を書き出した。

| name | fit split | 変更 | epoch（best） | test AUROC | ログ |
| --- | --- | --- | --- | --- | --- |
| `a_s42` | train_a | なし | 12（3） | 0.857 | `run_logs/lbh_a_s42.log` |
| `b_s42` | train_b | なし | 12（10） | 0.852 | `run_logs/lbh_b_s42.log` |
| `a_plant_s42` | train_a | マーカー（`planting.py`） | 12（3） | 0.790 | `run_logs/lbh_a_plant_s42.log` |
| `a_pacersub_s42` | train_a | ペースメーカー陽性を 10% に間引き（648→62） | 6（2） | 0.850 | `run_logs/lbh_a_pacersub_s42.log` |
| `a_yfdev_s42` | train_a | 65 歳未満女性 × デバイスの陽性を 10% に間引き（1142→101） | 6（4） | 0.853 | `run_logs/lbh_a_yfdev_s42.log` |
| `a_yfflip_s42` | train_a | 同じ部分集団の陽性 1142 枚をすべて陰性に反転 | 6（3） | 0.848 | `run_logs/lbh_a_yfflip_s42.log` |
| `a_piccsub_s42` | train_a | PICC に触れる陽性を 10% に間引き（818→74） | 6（5） | 0.855 | `run_logs/lbh_a_piccsub_s42.log` |
| `a_central_linesub_s42` | train_a | 中心静脈ラインの陽性を 10% に間引き（1583→165） | 6（1） | 0.850 | `run_logs/lbh_a_central_linesub_s42.log` |
| `a_sternotomysub_s42` | train_a | 胸骨切開・CABG の陽性を 10% に間引き（386→42） | 6（5） | 0.851 | `run_logs/lbh_a_sternotomysub_s42.log` |
| `a_surgical_clipsub_s42` | train_a | 手術クリップの陽性を 10% に間引き（445→45） | 6（5） | 0.850 | `run_logs/lbh_a_surgical_clipsub_s42.log` |
| `a_pacerflip_s42` | train_a | ペースメーカーの陽性 648 枚をすべて陰性に反転 | 6（3） | 0.841 | `run_logs/lbh_a_pacerflip_s42.log` |
| `a_central_lineflip_s42` | train_a | 中心静脈ラインの陽性 1583 枚をすべて陰性に反転 | 6（3） | 0.851 | `run_logs/lbh_a_central_lineflip_s42.log` |
| `a_piccflip_s42` | train_a | PICC の陽性 818 枚をすべて陰性に反転 | 6（1） | 0.852 | `run_logs/lbh_a_piccflip_s42.log` |
| `a_rand600d0_s42` / `a_rand600d1_s42` | train_a | 対照：要因と無関係に陽性を 600 枚除く（drop seed 0 / 1） | 6（3 / 1） | 0.854 / 0.852 | `run_logs/lbh_a_rand600d{0,1}_s42.log` |
| `a_rand1400d0_s42` | train_a | 対照：陽性を 1400 枚除く | 6（4） | 0.853 | `run_logs/lbh_a_rand1400d0_s42.log` |
| `a_s43` | train_a | 対照：変更なし、seed 43 | 6（5） | 0.855 | `run_logs/lbh_a_s43.log` |
| `isic_bcn_s42` | ISIC 2019 の BCN20000（`isic_sites.py`） | メラノーマの 2 値分類、`probes/train_stage1_isic.py` | 15（9） | `isic_test`（HAM10000 本体由来、公式 test ではない）0.791 | `run_logs/lbh_isic_bcn_s42.log` |
| `m01_s42` | race-known train 全体 | 既存 M01 s42 の特徴 cache を split_crossfit の行順に並べ替えただけ | — | 0.858 | — |

## e(x)（`probes/embed.py`、`cache/embed/<name>/`）

| name | モデル | 解像度 | split |
| --- | --- | --- | --- |
| `raddino` | microsoft/rad-dino（ローカル HF cache） | 518 | 全 4 split |
| `raddino_plant` | 同上、マーカーを描いた画像 | 518 | train_b / val / test |
| `dinov2` | facebook/dinov2-base（ローカル HF cache） | 224 | train_b / val / test |

## 凍結特徴の補正（`latent_codes.py`、`results/latent_codes/<name>/`）

名前の接頭辞で条件群を分ける。すべて 3 seed（`_s0`〜`_s2`）。

| 接頭辞 | 問い | 1段目 |
| --- | --- | --- |
| `cf_` | in-sample と cross-fit | a_s42・b_s42 |
| `dv_` | 振り分けの入力と λ | a_s42 |
| `lc_` | 振り分け係の容量と安定性 | a_s42 |
| `gd_` | test score を保存した主要条件（bootstrap 用） | a_s42・a_plant_s42 |
| `pl_` | マーカー植え込みの回収と修正（選択は train_b の患者 20% holdout） | a_plant_s42 |
| `ps_` | ペースメーカー間引きの回収と修正 | a_pacersub_s42 |
| `im_<1段目>_`（続き） | 上の 1 段目すべて（間引き・反転・対照）。条件は global・demo32・gmm32・router_meta・router_he の 3 seed | — |
| `im_<1段目>_` | 属性セル・GroupDRO・LHCF 型・GMM・振り分け係の改善比較（`sweep.py improve`、主要 4 条件は 5 seed） | a_s42・a_pacersub_s42・a_yfdev_s42・a_yfflip_s42 |

## backbone 側の補正（`probes/train_adapter.py`、`cache/adapter/<name>/`）

Spatial LoRA（stage3・stage4・fc、conv2、rank 4）、batch 128、AdamW（lr 1e-4、wd 1e-2）。

| name | 1段目 | fit split | 条件 | 備考 |
| --- | --- | --- | --- | --- |
| `ad_a_fita_meta` / `ad_a_fitb_meta` | a_s42 | A / B | 属性 | 6 epoch |
| `ad_a_fita_none` / `ad_a_fitb_none` | a_s42 | A / B | なし | 6 epoch |
| `ad_plant_fitb_none` / `ad_plant_fitb_meta` | a_plant_s42 | B | なし / 属性 | 6 epoch |
| `jt_plant_frozen_meta` / `jt_plant_joint_meta` | a_plant_s42 | B | 属性 | 4 epoch、凍結 / 同時学習 |
| `jt_pacer_frozen_router` / `jt_pacer_joint_router` | a_pacersub_s42 | B | 振り分け係（RAD-DINO） | 4 epoch、凍結 / 同時学習 |

## 監査（[latent_subgroups.md](reports/latent_subgroups.md)）

| 出力 | スクリプト | 対象 | ログ |
| --- | --- | --- | --- |
| `results/audit/<1段目>/` | `audit_run.py`（`--subgroup` で植え込みの答え合わせ） | 4 種類の 1 段目 | `run_logs/lbh_audit_run_<1段目>.log` |
| `results/audit/a_yfdev_s42_field/` | `audit_run.py --field`（振り分け係の群と補正量の分位） | a_yfdev_s42 | `run_logs/lbh_audit_yfdev_field.log` |
| `results/audit_corrected/im_<1段目>.csv` | `audit_corrected.py`（監査器 3 入力 × 3 seed） | `im_` の全 run と補正前 | `run_logs/lbh_audit_corrected_<1段目>.log` |
| `results/profile/a_s42_{he,d}.csv` | `profile_slices.py` | a_s42 | — |

`results/audit/a_s42/` の初期の出力（Platt・isotonic 較正、観測陽性数での群の選別）は、同じ場所に上書きした。
`a_yfdev_s42_field/` は帰無分布をポアソン近似にしていた版で、`families*.csv` の SD は他と比べられない（`planted.csv` は影響を受けない）。
| `results/eval_groups/im_<1段目>.csv` | `eval_groups.py`（粒度別の最小 AUROC・gap・BAcc） | a_s42・a_pacersub_s42・a_yfflip_s42 | — |
| `results/granularity/a_s42.csv` | `granularity.py`（最小 AUROC の bootstrap SD と勝者の呪い） | a_s42 | — |
| `results/diag_age/a_s42.csv` | `diag_age.py`（年齢差の構成による説明） | a_s42 | — |
| `results/dose_response/` | `dose_response.py`（植え込み要因と属性群の重なり・損害・回復） | 5 要因の間引き | `run_logs/lbh_dose.log` |
| `results/dose_response/beyond_null.csv` | `dose_response.py`（対照 4 本の帰無分布で標準化した損害と正味の回復） | 8 本の植え込み | — |
| `results/bgrid/{pacer,marker}.csv` | `bgrid.py`（監査セットの偏り × 大きさ × 9 手法 × 3 seed）、表示は `summarize_bgrid.py` | a_pacersub_s42・a_plant_s42 | `run_logs/lbh_bgrid_*.log` |
| `results/isic/discovery.csv` | `isic_discovery.py`（ISIC の施設間で属性群と潜在スライスの監査） | isic_bcn_s42 | — |
| `results/isic/grid.csv` | `isic_eval.py`（HAM の監査セットの大きさ × 8 手法 × 3 seed） | isic_bcn_s42 | `run_logs/lbh_isic_eval.log` |

マーカーの監査セットの素材として、`a_plant_s42` の train_b を全画像に印を付けた版と付けない版で書き出した
（`probes/export_marked.py`、`cache/stage1/a_plant_s42/train_b_mark{all,none}.npz`）。RAD-DINO の印あり版は
`probes/embed.py --mark-all`（`cache/embed/raddino_markall/train_b.npz`）。ISIC の e(x) は DINOv2（`probes/embed_isic.py`）。

## E2E の条件付け（[latent_routing.md](reports/latent_routing.md)）

条件の列は `latent_conditions.py` が `data/<dataset>/splits_latent/` に書く。学習は `projects/hypernet_e2e`（W&B 付き、
`study=latent_bias_hypernet`、seed 42〜44）。取り込んだ run は `runs/` に、予測は `cache/<run-id>_{val,test}.npz` にある。

| 出力 | 作るもの | 対象の run |
| --- | --- | --- |
| `results/heterogeneity.csv` | `heterogeneity.py`（凍結特徴での群別の頭・MLP の頭） | HAM・FairVLMed は DINOv2、ISIC は isic_bcn_s42、CheXpert は a_s42 |
| `results/ham10000_e2e/{summary,contrast}.csv` | `e2e_eval.py --dataset ham10000` | `resnet-ham10000`・`spatial-lora-ham10000`（10/06、別セッション）、`spatial-lora-ham10000-{const,random8,site,gmm8,gmm8lo,emb16,emb16lo,fc}`、`*-group-dro` 4 種 |
| `results/fairvlmed_e2e/{summary,contrast}.csv` | `e2e_eval.py --dataset fairvlmed` | `resnet-fairvlmed`・`spatial-lora-fairvlmed`（10/06）、`spatial-lora-fairvlmed-{const,random8,gmm8,gmm8lo,emb16}` |
| `results/<dataset>_e2e/condition_use.csv` | `condition_use.py`（条件を行の間で入れ替えたときの予測の変化） | 上の Spatial LoRA の run |

性別 × 年齢の GroupDRO の run は、test に sex の欠損があり群つきの datamodule で推論できないため、
test の予測だけ `predict_ungrouped.py` で群なしの datamodule に差し替えて書いた。

## 公平性手法の比較と振り分けの正の対照（[fairness_methods.md](reports/fairness_methods.md)）

学習は ResNet-50 または Spatial LoRA（`projects/hypernet_e2e`、W&B 付き、`study=latent_bias_hypernet`、seed 42〜44）。
fg の run は val の全体 AUROC と性別 × 年齢の最悪群 AUROC で 2 つの checkpoint を残し、SWAD の run は `checkpoints/swad.ckpt` を使う。

| 出力 | 作るもの | 対象の run |
| --- | --- | --- |
| `data/ham10000/splits_lhcf/`、`results/ham10000_e2e/lhcf_{bic,cohorts}.csv` | `lhcf_cohorts.py`（ERM の埋め込みの GMM、K は BIC。対照の列 `lhcf_random`・`label_group` も足す） | `resnet-ham10000-fg-erm` の seed 42 |
| `results/ham10000_e2e/fairness_{summary,contrast,pairs,weights}.csv` | `fairness_eval.py`（ERM との差、手法どうしの差、群の重みの偏り） | `resnet-ham10000-fg-*`：erm、dann-sex-age、{gdro,ug,resample,fairdro,minimaxauc,popauc,eo}-{sex-age,gmm8}、{wst,fis,fami}-{lhcf,sex-age}、swad、swadwst-{lhcf,sex-age,lhcfrand,label} |
| `results/ham10000_e2e/discovery_remedy{,_rank}.csv` | `discovery_to_remedy.py`（群の切り方ごとの発見の指標と DFR の効果） | `resnet-ham10000`、`resnet-ham10000-fg-erm` |
| `data/ham10000/images_rule{,_strong}/` | `rule_planting.py [--dose strong]`（GMM 8 のクラスタ 2・4 でマーカーの意味を反転） | — |
| `results/ham10000_e2e/rule_{summary,contrast,ceiling}{,_strong}.csv` | `rule_eval.py [--dose strong]`（AUROC と、マーカー on/off の反実仮想の効果） | `resnet-ham10000-rule`、`spatial-lora-ham10000-rule-{const,gmm8,random8}`、`spatial-lora-ham10000-rule-strong-{const,gmm8,random8}` |

SWAD の予測は `predict_ungrouped.py --swad`、最悪群で選んだ checkpoint の予測は `--worst` で書く（`fairness_eval.py` が足りないものを表示する）。

## Q2 の HAM10000 版：公平性損失 × モデル（`q2_ham10000.py`、2026-10-12）

`hypernet_e2e` の run（study=latent_bias_hypernet、W&B group `20261012-q2-ham10000-<preset>`）。1 段（ImageNet から E2E）、
30 epoch、全 epoch の重みを保存。train は病変 80%（`data/ham10000/splits_dual_holdout`、`data_pipeline/ham10000_dual_holdout_splits.py`）、
公平性損失の群の重みは残り 20% の held-out で更新。ResNet ERM はローカル GPU0、他の 15 run は ws11 の GPU 8〜4
（起動は `run_logs/q2_ham10000_{local,ws11_launch}_20261012.sh`）。checkpoint（1 run 約 3 GB）は ws11 に残し、
予測は ws11 で `export_onestage.py` が書き出したもの（`run_logs/q2_ham10000_ws11_export_20261012{,_retry1}.sh`）を回収した。

| preset | ResNet-50 の run（s42 / s43 / s44） | 属性条件付き Spatial LoRA の run（s42 / s43 / s44） |
| --- | --- | --- |
| `*_ham10000_dual_holdout_erm` | `20261012T101546Z-…-0f5b` / `20261012T102246Z-…-866e` / `20261012T102946Z-…-1d9f` | `20261012T102131Z-…-e112` / `20261012T102938Z-…-40a3` / `20261012T103736Z-…-22e4` |
| `*_ham10000_sex_age_eo_rate_dual_holdout` | `20261012T102140Z-…-2e02` / `20261012T103323Z-…-9eda` / `20261012T104502Z-…-a50b` | `20261012T102128Z-…-52da` / `20261012T103522Z-…-ad2b` / `20261012T104918Z-…-f379` |
| `*_ham10000_sex_age_minimax_auc_dual_holdout` | `20261012T102128Z-…-26c0` / `20261012T103255Z-…-76ef` / `20261012T104416Z-…-25ad` | `20261012T102124Z-…-83e3` / `20261012T103512Z-…-2f9e` / `20261012T104848Z-…-9898` |

| 出力 | 生成 | 入力 |
| --- | --- | --- |
| `cache/ham10000_q2/<model>_<loss>_s<seed>/` | `export_onestage.py`（epoch ごとの val・test の z1 − z0 と群の重み） | 上の 18 run |
| `results/q2_ham10000_{fairness_loss,threshold,auroc_decomposition,weights}.csv` | `q2_ham10000.py` | `cache/ham10000_q2/` |

学習中に記録された `val/auroc` は bf16 の softmax で同点が増えて低く出る（`MetricsLogger` が autocast の外で bf16 の
logits を渡すため。epoch 29 で 0.894 に対し fp32 では 0.905）。epoch は書き出した fp32 の予測から選び直している。
