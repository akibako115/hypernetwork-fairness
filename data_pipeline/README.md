# データ split の方針

`data/<dataset>/` 配下の split CSV が、どの単位で分割され、どの比率で、何を根拠にそうなっているかの
正本である。どの project も split CSV を読み取り専用の固定入力として扱うので、方針の変更はここと
生成コードの両方を更新する。

## 共通の契約

すべての split CSV は `projects/*/data/splits.py` の `validate_split_frame` が通る形を持つ。

| 列 | 内容 |
| --- | --- |
| `image` | 画像 directory からの相対 path。split 内で一意 |
| `target` | クラスの整数値 |
| `<属性>` | 整数コード（categorical）または実数（continuous） |
| `<属性>_missing` | 欠損フラグ。1 なら値は未定義 |

**欠損の判定は必ず `<属性>_missing` で行う。** 欠損行に入っている値はデータセットごとに違い、実在の
カテゴリと区別できない。

- CheXpert: 欠損 categorical は `0`。`race` なら `White`、`ap_pa` なら `AP` と同じ値になる
- ISIC 2019: 欠損 categorical は `-1`
- FairVLMed: 欠損 categorical は `-1`（配布 CSV の `unknown`）
- HAM10000: 欠損 categorical は `-1`（配布 CSV の `unknown` と空欄）

値そのものを見て欠損を判定すると、CheXpert では欠損が最大群へ静かに混ざる。消費側（`MetadataEncoder`、
属性 adversary、公平性指標、group DataModule）はいずれもフラグとペアで読む実装になっている。

## CheXpert

| 項目 | 内容 |
| --- | --- |
| 分割単位 | **患者**（`patientXXXXX`） |
| 比率 | 80/10/10（train 177,783 / val 22,900 / test 22,268） |
| 生成元 | 旧 repo `../fairness` の `src/create_cv_chexpert.py`。この repo では再生成しない |
| 場所 | `data/chexpert/splits/{train,val,test}.csv` |
| CV | **行わない。** 1 run が重く、2 クラス・22 万枚で評価セルが十分に大きいため |

生 metadata（`df_chexpert_plus_240401.csv`、223,462 行）から 511 行が除外されている。内訳は
`sex=Unknown` 279 行と `age` 欠損 278 行（重複あり）で、フラグで表現できない属性は行ごと落とす方針。
残る行については、全属性で欠損フラグの漏れが無く、非欠損行の符号も原文と一致することを確認済み。

### CheXpert の race 既知サンプルによる派生 split

`uv run python -m data_pipeline.chexpert_race_known_splits` で、元の `data/chexpert/splits/` を
保持したまま `data/chexpert/splits_race_known/` を作成する。`race_missing` が真の行だけを除外し、
所属 split・行順・既存の属性値は維持する。出力先が存在する場合は上書きしない。
入力・出力の変更には `--input-dir` / `--output-dir` を使う。

元の `race` は metadata encoder 用に残し、評価・GroupDRO 用の `race_group` を追加する。
元コード 0 / 2 / 3 を White(0) / Asian(1) / Black(2)、1 / 4 / 5 を Others(3) にまとめる。
`race_group_missing` は全行 false。入力 hash・使用前後の件数・クラス分布・対応表は
出力先の `manifest.json` に残す。期待する残存件数は train 157,852 / val 20,058 / test 19,617。
元データを保持することで、race 欠損群の補足評価と既存 run の再現に利用できる。

`uv run python -m data_pipeline.chexpert_crossfit_splits` で、race-known の train を患者単位で半分ずつに分けた
`data/chexpert/splits_crossfit/{train_a,train_b,val,test}.csv` を作る（seed 0、train_a 79,493 / train_b 78,359）。
1 段目を A だけで学習し、2 段目の補正を 1 段目が見ていない B で学習する cross-fit 用である。
val / test は race-known と同じ行の複製。使っていた study は `analysis/archive/latent_bias_hypernet/`（2026-10-17 に消した。snapshot/agent-era-20261016 ブランチに残る）。

## ISIC 2019

> **用途を限定している（2026-10-12）。** 皮膚画像の補足実験は HAM10000 にそろえ、下の `splits_fairness` と
> 5-fold CV は新しい実験では使わない。ISIC 2019 は、BCN20000 で学習して HAM10000 の部分に展開する施設間の
> 分析（`analysis/archive/latent_bias_hypernet/isic_sites.py`）にだけ使う。HAM10000 本体は ISIC 2019 に含まれるので、
> 一方の split で学習・fit したものを他方のデータに当てない。方針の正本は
> [docs/research-premises.md](../docs/research-premises.md)。

| 項目 | 内容 |
| --- | --- |
| 分割単位 | **病変**（`lesion_id`）。持たない 2,084 枚は 1 枚ずつ独立した group |
| 比率 | 80/10/10（train 20,291 / val 2,521 / test 2,519） |
| 層化 | クラス。`lesion_id` 内でラベルは割れないので group 層化が画像単位のクラス比を保つ |
| 生成 | `uv run python -m data_pipeline.isic2019_splits`（seed 12345） |
| 場所 | `data/isic2019/splits_fairness/{train,val,test}.csv` |

病変単位で分けるのは、同じ病変の別アングルが train と test に散ると test が楽観側へ歪むためで、
CheXpert が患者単位で分ける理由と同じ。比率 80/10/10 は
[MEDFAIR](https://arxiv.org/abs/2210.01725) が皮膚科データセット（HAM10000 / Fitzpatrick17k）で
使う値に合わせている。

`attribution` 列は元データセット（BCN20000 12,413 枚 / HAM10000 10,015 枚 / MSK 2,903 枚）を表す
共変量であり、**公平性の評価軸には入れない**。患者の属性ではないうえ、群ごとにクラス構成が違う
（MSK には BCC・AK・DF・VASC・SCC が 1 枚も無い）ため、群間 gap が色かぶりのショートカットではなく
有病率差を測ってしまう。ITA で肌色を推定する際に、色かぶりとの切り分けに使う想定で列だけ残す。
MSK のうち 2,074 枚は ISIC archive のダンプに isic_id が無く、`attribution_missing` が立つ。

## FairVLMed

[Harvard-FairVLMed](https://huggingface.co/datasets/harvardairobotics/FairVLMed)（FairCLIP, CVPR 2024）。
SLO 眼底画像 10,000 枚の glaucoma 2 クラス分類で、ライセンスは CC BY-NC-ND 4.0（非商用研究のみ）。

| 項目 | 内容 |
| --- | --- |
| 分割単位 | 患者。ただし 1 患者 1 枚なので、画像単位と同じ |
| 比率 | **配布元の `use` 列をそのまま使う**（train 7,000 / val 1,000 / test 2,000） |
| 生成 | `uv run python -m data_pipeline.fairvlmed_splits` |
| 場所 | `data/fairvlmed/splits/{train,val,test}.csv`、画像は `data/fairvlmed/images/*.png` |
| CV | 行わない |

配布 split を再分割しないのは、FairCLIP をはじめ先行研究の数値がすべてこの split 上のものだからである。
入力は `data/fairvlmed/` に置いた HF の `Dataset/dataset.zip` と `ReadMe/data_summary.csv`。
npz の `slo_fundus` を PNG へ展開し（`ImageDataset` は PIL で開くため）、その際に npz 内の
`glaucoma` / 属性の整数符号が CSV の符号化と一致することを全件で確かめる。

**npz の `language` は 2（Other）と -1（Unknown）が入れ替わっている。** CSV は other 76 件・unknown
498 件で、論文の「0.8% が other、5.0% が unknown」と一致するので CSV を正とする。npz を直接読む
FairCLIP 公式実装は `language == -1` を unknown として評価から外すが、実際に外れているのは other の
76 件で、unknown の 498 件が Others 群として評価されている。先行研究の language 軸の数値と
比べるときはこの差に注意する。

属性は gender / race（Asian・Black・White）/ ethnicity / language / maritalstatus と age で、
欠損は age 以外にあり `unknown` で表される。fairness の評価軸は FairCLIP の 4 軸（race / gender /
ethnicity / language）に `age_group_65` を加えたもので、maritalstatus はモデル入力にだけ使う。

test の language は other が 11 枚（陰性 5 / 陽性 6）、spanish が 31 枚（陰性 9 / 陽性 22）しかない。
後述の最小セル枚数 20 を当てると、other は全セル、spanish は陰性側（TNR / FPR）のセルが外れる。
**ただしこの下限は事後分析で適用する規則で、学習中の `FairnessMetricsCallback` には実装されていない。**
callback がログする language 軸の gap は 11 枚の other 群を含んだ max−min なので、参考値として扱う。

split CSV には、属性ではないフラグ列を 2 つ持たせる。どちらもモデル入力にも評価軸にも入れず、
事後分析で評価対象を絞るために使う。

| 列 | 内容 | test |
| --- | --- | --- |
| `image_blank` | 輝度 10 未満の画素が 95% 以上を占める、ほぼ黒い画像 | 112 / 2,000 |
| `fairclip_eval_subset` | FairCLIP 公式実装が val / test の評価に残す行 | 1,883 / 2,000 |

`image_blank` は全体で 552 枚（5.5%）あり、配布 zip の JPEG も同じ暗さなので変換の問題ではない。
目視では大半が撮影失敗で、残りもコントラストを上げてようやく眼底が見える露出不足である。率は
Black 12.3% / Asian 9.2% / White 3.8% と race で大きく違い、Black の陽性では 14.4% に達する。
画像から判定できない例が特定の群の陽性に偏るので、群間 gap の一部は画質の差を測ってしまう。
FairCLIP はこれらを除かずに評価しているため行は残し、除いた評価を感度分析として並べる。
閾値の境界に明確な谷は無く（90% 以上なら 688 枚）、操作的な定義である。blank を除いても平均輝度は
White 40 / Black 22 / Asian 23（0–255）と race で違う。

`fairclip_eval_subset` は、先行研究の数値と突き合わせるための評価対象である。FairCLIP 公式実装は
npz の race / gender / ethnicity / language のいずれかが -1 の行を、群別だけでなく全体 AUC からも外す。
npz の language は入れ替わっているので、外れるのは ethnicity の unknown と language の other になる。

## HAM10000

[HAM10000](https://doi.org/10.7910/DVN/DBW86T)（Harvard Dataverse v4.0）。ダーモスコピー画像 10,015 枚
（7,470 病変）と、ISIC2018 Task3 の公式 test 1,512 枚。ライセンスは CC BY-NC 4.0（非商用のみ）。

| 項目 | 内容 |
| --- | --- |
| target | **悪性 2 値**（mel / akiec = 1、MEDFAIR と同じ）。7 クラスは `dx` 列に残す |
| 分割単位 | 病変（`lesion_id`）。全画像が持つ |
| 比率 | 本体を train/val 85/15（train 8,457 / val 1,491）。test は公式 test 1,511 |
| 除外 | train/val から sex か age が欠損した 67 枚（train 62 / val 5）。test は全行を残す |
| 層化 | 7 クラスの `dx`。`lesion_id` 内で dx は割れない |
| 生成 | `uv run python -m data_pipeline.ham10000_splits`（seed 12345） |
| 場所 | `data/ham10000/splits/{train,val,test}.csv`、画像は `data/ham10000/images/*.jpg` |
| CV | 行わない |

入力は `data/ham10000/raw/` に置いた Dataverse の配布物で、画像 zip 3 本を `images/` 直下へ展開する。
本体の 10,015 枚は ISIC 2019 Training にもすべて含まれる（`attribution` の HAM10000 群）。ISIC 2019 の
代わりにこちらを使う理由は、ラベル付きの公式 test を外部 test として使えることにある。公式 test は
本体と画像 ID も病変も重ならず、病変単位の漏洩が起こらない。test の `ISIC_0035068` は GroundTruth に
あるが画像 zip に無く、dx 以外の metadata も空なので外している。

**target を 2 値にするのは、7 クラスでは評価セルが足りないため。** 公式 test を属性で割ると df・vasc は
群あたり 1 桁になり、後述の最小セル枚数 20 を割る。2 値なら最小セルは sex 軸で female の悪性 69 枚、
age 軸で 65 歳未満の悪性 94 枚ある。

2 値の定義は [MEDFAIR](https://arxiv.org/abs/2210.01725) の公式前処理（`notebooks/HAM10000-example.ipynb`）に
合わせ、mel と akiec を悪性、bcc を含む残り 5 クラスを良性とする。MEDFAIR は Maron et al. (2019) に従って
bcc を良性側に置いている。臨床的には bcc も悪性腫瘍だが、先行研究との比較可能性を優先した。悪性の比率は
train 14.5% / test 14.2%。分割は MEDFAIR と異なる（MEDFAIR は本体を病変単位で 80/10/10）ため、数値を
直接は比べられない。

属性は sex / localization（categorical）と age（continuous）。fairness の評価軸は sex と
`age_group_65` で、CheXpert・ISIC 2019 と同じく 65 歳で切る。HAM10000 の age は 5 歳刻みなので、
MEDFAIR の 2 区分（論文では 60 歳、コードでは 60 歳を若年側に入れる `≤60 / ≥61`）とも同じ群になる。肌色・人種の列は無く、撮影地もオーストリアとオーストラリアに
限られる。

欠損は本体では少ない（sex 57 / age 57 / localization 234、うち 47 件は sex と age が両方欠ける）。
**train/val からは sex か age が欠損した行を外す。** どちらも fairness の評価軸と Group DRO の group を
決める属性で、欠損行があると group が決まらない（`group_datamodule` は例外で止まる）。欠損を独立の
group にすると数十行の group を DRO が過大に重み付けし、ERM の run でだけ残すと手法間で学習データが
揃わない。病変の train/val 割り当ては全行で決めてから外すので、外す行の有無で他の病変の所属は動かない。
localization は評価軸ではなくモデル入力にだけ使うので、欠損フラグのまま残す。

| split | sex か age の欠損 | 扱い |
| --- | --- | --- |
| train | 62 / 8,519（0.7%） | 外す |
| val | 5 / 1,496（0.3%） | 外す |
| test | 294 / 1,511（19.5%） | 残す。全体の性能は 1,511 枚、fairness 指標は欠損フラグで外れた残り 1,217 枚で計算する |

**test は `dataset=external`（HAM10000 外の施設、316 枚）に欠損が集中し**、age か sex を欠く 294 枚の
うち 284 枚がここに属する。fairness 指標は実質 HAM10000 と同じ施設の画像に限られる。

**属性で条件付けたモデルは、external の欠損行で崩れる。** 欠損年齢は配布 CSV で 0 歳として入り、標準化すると
約 −3 SD の値になる。train/val から欠損行を外しているので、モデルはこの入力も、立った欠損フラグも学習中に見ない。
2026-10-07 の ERM の比較（旧 `ham10000_architecture_erm`）では、external だけの AUROC で Spatial LoRA・HyperAdapt が
ResNet-50 を下回った（0.822・0.799 対 0.846）。属性を使わない ResNet と比べると、全 1,511 枚の全体 AUROC には
モデル構成の差と欠損入力の扱いの差が混ざるので、**モデル間の主比較は属性の揃った 1,217 枚で行う**。

**患者単位の分離は保証できない。** 患者 ID が公開されておらず、ISIC archive でも全画像の `patient_id` が
空である。同一患者の別病変が split をまたぐ可能性は残る。確認できたのは、split 間で `lesion_id` の共有が
無いこと（archive 側の lesion ID でも同じ）と、dHash で split をまたぐ画像重複が無いこと（距離 10 以下の
6 組は目視ですべて別病変）である。`dataset`（撮影元）と `dx_type`（診断の根拠: histo / follow_up / consensus /
confocal）は共変量として列だけ残し、評価軸には入れない。

`age=0` は本体に 39 件、test に 6 件ある。本体の 39 件はすべて vidir_modern 由来の良性
（nv 30 / bkl 5 / vasc 4）で、先天性母斑や乳児血管腫と考えて矛盾しないので、ISIC 2019 と同じく実在の
値として扱う。

## CV 戦略

**ISIC 2019 のみ 5-fold CV を行う。CheXpert は単一 split のまま。**

> ISIC 2019 の用途を限定したので（上の ISIC 2019 節）、この CV は新しい実験では使わない。過去の判断の記録として残す。

```bash
uv run python -m data_pipeline.isic2019_splits --cv
# -> data/isic2019/splits_fairness_cv/fold{0..4}/{train,val,test}.csv

uv run python -m projects.hypernet_e2e.run experiment=spatial_lora_isic2019_cv fold=0
```

| 項目 | 内容 |
| --- | --- |
| fold 数 | 5。[Gessert et al.](https://arxiv.org/abs/1910.03910)（ISIC 2019 チャレンジ優勝）と同じ |
| 分割単位 | 単一 split と同じ病変単位。Gessert et al. も *"all images of the same lesion are in the same fold"* としている |
| fold 内の比率 | 60/20/20（train 3 fold / val 1 fold / test 1 fold）。ISIC 文献の 6:2:2 に一致し、追加の乱数を使わずに決まる |
| 割り当て | fold k は test に fold k、val に fold k+1（巡回）、train に残り 3 fold |
| 評価 | 5 run の test 結果をプールする。全 25,331 枚がちょうど 1 回ずつ test に回る |

### なぜ ISIC だけ CV が要るか

稀少クラスの絶対数が足りない。単一 split の test（10%）では、属性で群に割った時点で群あたり 10 枚
程度にしかならない。

```
              全体    現 test(10%)   test 20%   5-fold CV でプール
DF             239         24           48           239
VASC           253         25           51           253
SCC            628         62          126           628
```

比率を 20% に変えても DF は 48 枚で、sex で割れば片側 20 枚台にしかならない。**比率では解けず、
全画像を 1 回ずつ test に回すしかない。** CV でプールすると、後述の閾値 20 の下で 3 軸すべて
（sex / age_group_65 / anatom_site_general）の全 8 クラスが評価対象に入る（単一 split では 5/8）。

CheXpert に CV を行わないのは、2 クラス・test 22,268 枚で最小セルが 391 枚（`race` の Native
American を除く）あり、評価セルの不足が起きないため。学習コストに見合わない。

## 群別指標の最小セル枚数

公平性指標の群間 max−min は、群が小さいほど上振れする統計的にバイアスのある推定量である
（[Lum, Zhang & Bower, FAccT 2022](https://dl.acm.org/doi/10.1145/3531146.3533105)）。実際、群間の
不公平がゼロのモデルで ISIC の部位軸を測ると Eopp1 が 0.44 出る（sex 軸は 0.087）。

そこで **rate を推定するセルの枚数に下限 20 を課す**。TPR は当該クラスの陽性数、TNR / FPR は陰性数を
見る。20 という値は [FRAME](https://arxiv.org/html/2608.25981) が prespecify している
*"a subgroup is evaluable when it contains at least 20 labeled images"* から取り、評価単位を群から
セルへ読み替えている（8 クラスの one-vs-rest では、群が 500 枚あっても特定クラスの陽性が 2 枚に
なりうるため）。論文に書く際はこの読み替えを明示する。

除外の判定は test split のセル枚数だけで決まり、モデルの予測に依存しない。したがって手法間の
比較は常に同一のセット上で行われる。
