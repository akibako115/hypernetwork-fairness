# latent_bias_hypernet

> **2026-10-13 注記（公平性損失の実装修正）**：[修正記録](../../../docs/reviews/20261013_fairness_objective_fixes.md)。[fairness_methods.md](reports/fairness_methods.md) と Q2 のうち、次の行は手法の効果として読まない。
> GroupDRO（自然抽出で、バッチに出ない群の q が縮んでいた）、EO 制約（旧 smoothstep 代理制約。
> `probes/train_adapter_fair.py` の EO も同じ）、FIS（c が混合比として働いていなかった）、DANN（adversary の head が
> 属性を読めていなかった）、FairDRO（ρ=1 は 4 群・8 群で負の重みになる値）。
> minimax AUC・PopAUC・均等抽出・再重み付け・ℓ_wst・FaMI・SWAD は影響を受けない。
> したがって「11 種回しても最悪群 AUROC が動かない」は、影響を受けない手法についてだけ言える。

2026-10-08開始。固定した1段目のモデルに hypernetwork の補正（ΔW_k, Δb_k）を足し、
「モデルのバイアスを hypernetwork 側に溜めて、潜在的なバイアスやサブグループを発見・改善する」ための
設計を実験で決める。

## 問い

1. **どこで溜めるか。** 1段目が学習した split（in-sample）で補正を学習すると、残差は記憶済みの雑音が主になり、
   補正は val で悪化する、という仮説を検証する。見ていない split（cross-fit）で学習すれば汎化する補正が溜まるか。
2. **発見できるか。** グループを事前に決めず、タスクと無関係な表現 e(x) から振り分け係が決めるとき、
   モデルに渡していない要因（植え込んだマーカー、レポート種別、追跡撮影、デバイス）を回収できるか。
   k-means（e(x) だけで分ける）、既知の8群、属性入力の hypernet と比べる。
3. **改善するか。** グループ別の補正は、全体の再較正（K=1）を超えて test の全体・worst-group 性能を上げるか。
4. **どの形の補正が要るか。** 切片だけ（rank 0）と重み方向を含む補正（rank 4）の差。
5. **属性の掛け合わせで偏りが隠れるか。** 事前に決めた属性セルで監査すると、モデル内部の偏りが見えなくなるか。
   属性を使わない群（LHCF の GMM、残差の監査器、振り分け係）で発見・改善できるか。

## 設計

- split：`data/chexpert/splits_crossfit/`（train を患者単位で A / B に二分、val / test は race-known と同じ）。
  生成は `data_pipeline/chexpert_crossfit_splits.py`。
- 1段目：ImageNet 初期化の ResNet-50 を A だけで ERM 学習（`probes/train_stage1.py`）。
  植え込み版は A と val でマーカーを陽性と相関させ、B と test では独立に付ける（`planting.py`）。
- e(x)：RAD-DINO（胸部X線の自己教師あり、ラベル不使用）と DINOv2（`probes/embed.py`）。
- 凍結特徴の補正：`latent_codes.py`（head への補正）、まとめて回すのは `sweep.py`、表は `summarize.py`。
- backbone 側の補正：`probes/train_adapter.py`（hypernet_e2e と同じ Spatial LoRA を stage3・stage4・fc に付ける）。
- 答え合わせの要因：`context_labels.py`（レポート種別・追跡撮影・デバイス・ペースメーカー）。
- 同時学習の対照：`probes/train_adapter.py --unfreeze-backbone` と、LoRA を外して評価する `probes/eval_adapter_off.py`。
- 補助：`controls.py`（e(x) を線形に足す対照）、`stability.py`（seed 間の ARI）、`stats.py`（患者単位 paired bootstrap）。
- 属性セルと潜在スライスの監査：物差しと群の切り方は `audit.py`、発見側は `audit_run.py`、補正後のモデルを
  独立な監査器で測るのは `audit_corrected.py`（表示は `summarize_corrected.py`）、スライスの中身は `profile_slices.py`。
  部分集団の植え込み規則は `planting.HIDDEN_SUBGROUPS`。

## 結論

[発見の設計の記録](reports/findings.md)。1段目を凍結し、1段目が見ていないデータ（cross-fit）で補正を学習し、
振り分けには h(x) と e(x) の両方を入れ、読むのはグループ番号ではなく補正の向きと量にする。
植え込みバイアス（マーカー、ペースメーカー陽性の間引き）はこの設計で発見・修正できた。
in-sample では補正が1〜3 epoch で止まり、同時学習では補正が backbone 側へ移る。
実データの CheXpert No Finding では大きな潜在バイアスは見つからず、改善は全体 AUROC で 0.01 未満だった。

[属性の掛け合わせと潜在部分集団](reports/latent_subgroups.md)。属性セルで監査すると、実データの 1 段目の偏りは
標本誤差と区別できないが、属性を使わない監査器は潜在スライスの間に log O/E の SD 0.24 の偏りを見つける。
属性に無い部分集団（ペースメーカー）の偏りは属性セルでは消え、GMM クラスタや振り分け係では見える。
改善は [h; e] で振り分ける hypernet が一貫して最もよく、LHCF の minimax（GroupDRO）はほとんど効かない。

[監査セットの条件と属性評価への波及](reports/audit_set_and_transfer.md)。監査セットによる事後補正には、偏りの少ない
データが十分に要り、その量はショートカット（1,000 枚）と学び損ねた部分集団（2 万枚以上）で大きく違う。監査セットに
偏りが残ると、柔軟な補正ほど偏りを強める。潜在的な偏りを直した結果が属性評価に表れるのは、偏りが属性群に集中している
ときだけで、手法は選ばない。ISIC 2019 の施設間でも、属性で切ると偏りは見えず潜在スライスでは見えたが、補正は
最終層の作り直しが最もよかった。論文としての位置づけは [paper_framing.md](reports/paper_framing.md)。

[潜在空間で振り分ける条件付きモデル](reports/latent_routing.md)。属性の代わりに潜在の部分群（DINOv2 の GMM クラスタ、
ラベル直交のクラスタ、埋め込み）で Spatial LoRA を E2E に条件付けても、HAM10000・FairVLMed のどちらでも、定数で条件付けた
同じ構成と区別できなかった（3 seed）。学習後に条件を入れ替えても予測はほぼ動かず、画像から計算できる条件は使われない。
属性の利得は分類層だけに渡しても同じだけ出て、属性が欠けた施設ではそのほうがよい（external 0.871 対 0.822）。

[公平性手法の比較と、振り分けの正の対照](reports/fairness_methods.md)。性別 × 年齢・GMM 8・LHCF の隠れコホートを群にして
公平性手法を 11 種回しても、性別 × 年齢の最悪群 AUROC は ERM から有意に動かない。唯一効いた SWAD ＋ LHCF の ℓ_wst は、
群をクラスに替えても同じ利得が出る（コホートが悪性度で切れ、難しさの重み付けとして働く）。規則の食い違いを植え込んだ
正の対照では、反転の境界を含む区分で条件付けたモデルだけが少数側の規則を学び、手がかりが強ければ少数側の AUROC を
+0.094 上げた。振り分けの仕組みは働き、実データに規則の食い違いがないことが効かない理由と読める。
