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

## ISIC 2019

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

## CV 戦略

**ISIC 2019 のみ 5-fold CV を行う。CheXpert は単一 split のまま。**

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
