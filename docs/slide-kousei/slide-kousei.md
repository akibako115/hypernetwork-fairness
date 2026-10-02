## 現状での問題点をまとめて報告するためのスライド

1. class weight と fairness の切り離しの問題

- 先行研究では class weight について言及しているものがほとんどない
- サブグループごとの重み付けや出力・分布を合わせることのみにフォーカスしている
- 上記の具体例：以下の中から適切なものを選んで例として提示

"""
3 つのパターン
パターン A：群は属性だけ、損失はクラスを混ぜたまま → 陽性率が群の難しさに混ざる

手法	根拠
MEDFAIR の GroupDRO	重みなし BCE の群平均で q を更新する。step size 0.01 で固定。補正なし
Zhang et al. CHIL 2022 の GroupDRO	同上。CE で、class weight なし
FIS（FairVision）	群の重みを、陽性と陰性を混ぜた損失分布の Sinkhorn 距離から計算する
LHCF	群のリスク R_k は単純な群平均損失。群の陽性率は 0〜66% とばらついているのに、補正はない
FairDi の第 0 段と生徒モデル	FIS を使うので、FIS と同じ
パターン B：群を「属性 × ラベル」のセルにする → 群ごとに切片がずれる

手法	根拠
Yang et al. Nature Medicine 2024 の GroupDRO、ReSample、DFR	g = y·A + a のセル単位。ReSample では、どの群もバッチ内の陽性率が 50% になる。論文本文は "equalize group size" と書いているが、コードはセル単位
MEDFAIR の resampling（balanced モード）	セル単位で均等に抽出する
FairDi の教師モデル	各群の教師が、自群の陽性率に合わせた切片を学習する。生徒はこれに λ=0.95 の KL で合わせるので、群ごとのずれがそのまま移る（コードがないので論文からの推論）
パターン C：正解クラスで条件付ける → 陽性率の違いは定式化の段階で消える

手法	根拠
Zhang et al. の MMDMatch / MeanMatch	y=0 と y=1 のそれぞれで、群のスコア分布を全体の分布に合わせる
Zhang et al. の adversarial、Yang et al. の CDANN	adversary の入力に y を入れる。CDANN は adversary の損失のクラスも均等化している
Zhang et al. の FairALM	TPR と FPR の surrogate を、群ごとに全体の値へ合わせる
CFair	全体共通の class weight（balanced error）と、Y ごとに別の adversary
"""


2. group dro を例とした具体的な問題

- サブグループを考えた際に，それぞれ陽性率が明確に異なる．
  1. サブグループの損失を一律で global な class weight で適用した場合
    - 陽性率の高いサブグループほど高いクラス重みが多くのサンプルに割り当てられ，損失が大きくなる．これによって，識別の難しさ・バイアスではなく陽性率の違いに大きく依存した学習調整になってしまう
  2. サブグループごとに class weight を用意して適用した場合
    - サブグループごとに最適化している目的関数が異なる．結果として，class weight の違いによって logit のスコア分布がずれてしまい，global な識別性能を大きく低下させる．

- 解決案としては　「群の難しさをサブグループごとの class weight で測る」後に「最適化は global な class weight で行う」 ことで上記の問題は解決できそう
  - しかし，「Hypernetwork に公平性損失を割り当てる」 という主目的に対して，class weight を考慮した試行錯誤を行うべきかは不明．
