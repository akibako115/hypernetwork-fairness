# hyperadapt_two_stage_vs_e2e

HyperAdapt を、既学習 ResNet を凍結して適応する two-stage 条件と、ImageNet 初期化から全パラメータを更新する single-stage E2E 条件で比較する。

両条件は CheXpert の train split から算出する inverse class weight を用い、seed 43 に固定する。two-stage 条件の親は `20260922T063725Z-resnet-chexpert-s43-efcd` の最良 validation AUROC checkpoint とする。

公平性と性能の結論は、両 run が完走して artifact を取り込んだ後に記す。
