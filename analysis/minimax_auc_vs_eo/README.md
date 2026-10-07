# Minimax AUCとEO rate制約を同じ条件で検証する

2026-10-06。検証する公平性手法としてMinimax AUCとrate制約によるEqualized Odds（EO）を選んだ。
このstudyでは、群平均CEに含まれる陽性率の影響を避け、順位性能と共通閾値でのエラー率をそれぞれ改善できるか調べる。
現時点は実装・実験の詳細計画を作成した段階。新しい目的関数の実装・学習・結果の収集はまだ行っていない。

## 詳細計画

- [実装計画](implementation_plan.md)：年齢2群の4群ペアrisk、4本のEO制約、samplerと損失補正、更新順序、checkpointとテスト。
- [実験計画](experiment_plan.md)：同seedのERMからの継続学習、8条件の対照、探索・3seed確認、validation選択とtest評価。

主比較は原著の機構に沿って実装する。Minimaxは群内・群間の双方を扱い、EOはhard rateで制約側を更新する。
今回設計するEO proxy、AdamW、ERM初期化、単一checkpoint選択は原著からの変更として記録する。
EO marginは主比較後の追加候補とする。

## 比較する問いを分ける

| 条件 | 検証する問い | 主に見る指標 |
| --- | --- | --- |
| ERM | 現行の基準性能はどこか | 全体・各群AUROC、TPR、FPR |
| global class weightのGroup DRO | 群CEによる適応重みと比べて改善するか | 上記に加え、群別q・クラス別CE |
| Minimax AUC | 最悪の群ペアの順位性能を改善できるか | 群内worst AUROC、群間を含む最小AUC、全体AUROC |
| EO rate制約 | 共通閾値でTPR・FPR gapを抑えられるか | hard TPR/FPR gap、制約違反、worst TPR/TNR、全体AUROC |
| FairDRO（2026-10-07追加） | クラスごとの群重みを0-1誤りで更新する先行手法とEO制約の差 | EO rate制約と同じ。加えてクラス別q |

単純対照として群の等重み学習、動作点の参照としてERMのvalidationで決めた群別閾値も比較する。
群の等重みだけでは、各群内の陽性率によるCEの混合は残る。

## 原著と簡易版を混ぜない

Minimax AUCの第一案は[Yang et al., AAAI 2023](https://ojs.aaai.org/index.php/AAAI/article/view/26405)に沿い、
同じ群の陽性・陰性だけでなく、異なる群の陽性・陰性も扱う。初回の年齢2群では4種類の群ペアになる。
代理損失、モデル更新、群ペア重みの更新、初期化を原著と照合してから実装条件を確定する。
CEを加える版や群内ペアだけの版は、原著再現とは別の実験条件として記録する。

EOは[Cotter et al., JMLR 2019](https://jmlr.org/papers/v20/18-616.html)のproxy-Lagrangianを参照し、
モデル更新に微分可能な代理制約、制約側の更新にhard rateを使う。
今回の制約は `TPR gap <= epsilon_pos` と `FPR gap <= epsilon_neg`。
原著の確率的なモデル混合と、単一checkpointを選ぶ実用版を区別して評価する。
単純なsigmoid平均差のpenaltyを使う場合は、soft EO penaltyとして別名にする。

## 初回はResNetと既存のrace-known条件に限定する

初回の群定義は年齢のみ、65歳未満／65歳以上の2群とする。
対象はCheXpert No Finding、`splits_race_known`。群定義だけを変更し、既存のsplit・親ERMは維持する。
raceは公平性学習の群定義に使わない。年齢×raceの8群への展開は、年齢2群の結果を確認した後の別段階とする。
主比較はResNet-50、batch size 128、AdamW（lr 1e-4、weight decay 1e-2）、seed 42・43・44。
同seedの既存ERM checkpointから全層を追加30 epoch学習し、全対照の親と追加予算をそろえる。
まずseed 42で探索し、選んだ設定を43・44へ展開する。原著のAUCMax初期化は追加実験として分ける。

CEを使う条件のclass weightは現行のglobal inverseを基本とし、公平性の条件付き統計には掛けない。
純粋なAUC目的の条件にはclass weightを使わない。
新手法は年齢2群×2クラスの4セルを確保する層化batchを使い、同じbatch上の分類損失にimportance補正を入れる。
samplingの効果と目的関数の効果を区別する対照を用意する。
空セル率、群×クラス件数、有効な群ペア数、群ペア重み、hard/代理制約の値を記録する。

探索はvalidationに限定する。更新率・proxy幅の探索表、EOの主許容差0.10、AUROCの許容低下幅0.005を詳細計画に定めた。
これらは今回の設計値であり、原著の推奨値ではない。許容範囲に入る候補から、Minimax AUCは目的に対応する最小AUC、
EOは制約の実行可能性と分類性能で選ぶ。従来のbest val AUROCで選んだ結果も比較用に残す。
最終選定後にtestを評価し、患者単位のpaired bootstrapで差の不確実性を示す。
W&Bのgroupはseedを含めず、手法とハイパーパラメータが同じrunだけを束ねる。

## 今日の機構診断を比較設計に反映する

[既存の機構診断](../fairness_mechanism_probe/reports/mechanism_findings.md)では、
凍結特徴上のCE＋群内worst-rankのworst AUROCはERM比で約0.004の改善だった。
これはMinimax AUC原著の検証ではなく、今回の群間ペア・表現更新の効果は未確認である。
この診断は交差8群での結果であり、今回の年齢2群のworst指標とは分ける。初回は各年齢群の指標を併記する。

No Findingでは陽性が「所見なし」を意味する。EO評価ではTPR側だけでなく、所見ありを所見なしと判定するFPR側を明示する。
gapが縮んでもworst TNRや判別力が下がっていれば、性能改善とは判断しない。
群別閾値による参照と比べ、学習が順位を変えたのか、主にスコアの動作点を変えたのかを調べる。
レポート種別によるラベルの構成差は診断上の仮説として扱い、今回の公平性損失が原因を解消すると前提にしない。

## 起動前に確定する項目

- 親ERMの取り込みと実行先でのcheckpoint path。3seedのrun-id・checkpoint hash・split hashは照合済み。
- 実装・preflight・dry-runで確認した解決済み設定。
- 動作確認の時間・メモリ・disk実測、実行先とGPU、既存プロセスへの影響。

起動条件がそろったらprojectの公開入口でdry-runとpreflightを行い、解決済み設定を確認する。
対象runは[runs.md](runs.md)に記録する。

実装後の入口・検証状況は [implementation_status.md](implementation_status.md)、
原著・著者repoとの対応と変更点は [source_alignment.md](source_alignment.md) に記録した。

1 stage（ImageNet 初期化から学習）への適用は [conditional_fairness_one_stage](../conditional_fairness_one_stage/README.md) が持つ。
