// slide-kousei.md をもとに slide-kousei.pptx を生成する。
// 先に figures.py で figures/ に図を書き出してから実行する: node build.js
const path = require('path');
const pptxgen = require('pptxgenjs');
const pres = new pptxgen();
pres.layout = 'LAYOUT_16x9'; // 10 x 5.625
const F = 'Yu Gothic';
const INK = '222222', MUTED = '666666', RULE = 'BBBBBB';
const THEME = '1F3B57', THEME_TINT = 'E8EEF3';
const DECK = '現状の問題点：class weight と公平性の切り離し';
const OUT = process.argv[2] || path.join(__dirname, 'slide-kousei.pptx');

let page = 0;
function slide(t) {
  const s = pres.addSlide(); s.background = { color: 'FFFFFF' }; page++;
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.8, fill: { color: THEME }, line: { color: THEME } });
  s.addText(t, { x: 0.5, y: 0, w: 9, h: 0.8, fontFace: F, fontSize: 20, bold: true, color: 'FFFFFF', valign: 'middle', margin: 0, isTextBox: true });
  s.addShape(pres.shapes.LINE, { x: 0.5, y: 5.15, w: 9, h: 0, line: { color: RULE, width: 0.5 } });
  s.addText(DECK, { x: 0.5, y: 5.2, w: 7, h: 0.25, fontFace: F, fontSize: 9, color: MUTED, margin: 0, isTextBox: true });
  s.addText(String(page), { x: 9.0, y: 5.2, w: 0.5, h: 0.25, fontFace: F, fontSize: 9, color: MUTED, align: 'right', margin: 0, isTextBox: true });
  return s;
}
// items: [text, level] ; level 0 = bullet, 1 = sub-bullet
function bullets(s, items, y = 1.1, h = 3.9, w = 9) {
  const runs = items.map(([t, lv = 0], i) => ({
    text: t,
    options: { bullet: lv ? { indent: 14 } : { indent: 14 }, indentLevel: lv, fontSize: lv ? 13 : 14,
      color: lv ? '444444' : INK, paraSpaceAfter: 6, breakLine: i < items.length - 1 },
  }));
  s.addText(runs, { x: 0.5, y, w, h, fontFace: F, valign: 'top', margin: 0, isTextBox: true });
}

// 1. Title
{
  const s = pres.addSlide(); s.background = { color: 'FFFFFF' }; page++;
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 1.8, w: 10, h: 1.6, fill: { color: THEME }, line: { color: THEME } });
  s.addText(DECK, { x: 0.5, y: 2.0, w: 9, h: 0.65, fontFace: F, fontSize: 24, bold: true, color: 'FFFFFF', valign: 'middle', margin: 0, isTextBox: true });
  s.addText('先行研究の整理と GroupDRO での具体例', { x: 0.5, y: 2.7, w: 9, h: 0.45, fontFace: F, fontSize: 14, color: 'D6E0E8', valign: 'middle', margin: 0, isTextBox: true });
}

// 2. Problem 1
{
  const s = slide('1. class weight と fairness の切り離しの問題');
  bullets(s, [
    ['先行研究では class weight について言及しているものがほとんどない'],
    ['サブグループごとの重み付けや、出力・分布を合わせることにのみフォーカスしている'],
    ['クラスの扱い方で、先行研究は次の 3 パターンに分かれる'],
    ['A：群は属性だけ、損失はクラスを混ぜたまま → 陽性率が群の難しさに混ざる', 1],
    ['B：群を「属性 × ラベル」のセルにする → 群ごとに切片がずれる', 1],
    ['C：正解クラスで条件付ける → 陽性率の違いは定式化の段階で消える', 1],
  ]);
}

// 3. Examples table
{
  const s = slide('パターン別の代表例');
  const H = { bold: true, color: 'FFFFFF', fill: { color: THEME } };
  const rows = [
    ['A', 'MEDFAIR の GroupDRO', '重みなし BCE の群平均で q を更新する。step size 0.01 で固定。補正なし'],
    ['A', 'FIS（FairVision）', '群の重みを、陽性と陰性を混ぜた損失分布の Sinkhorn 距離から計算する'],
    ['A', 'LHCF', '群のリスク R_k は単純な群平均損失。群の陽性率は 0〜66% とばらつくが補正なし'],
    ['B', 'Yang et al. (Nat. Med. 2024)\nGroupDRO / ReSample / DFR', 'g = y·A + a のセル単位。ReSample では、どの群もバッチ内の陽性率が 50% になる'],
    ['B', 'MEDFAIR の resampling（balanced）', 'セル単位で均等に抽出する'],
    ['C', 'Zhang et al. の MMDMatch / MeanMatch', 'y=0 と y=1 のそれぞれで、群のスコア分布を全体の分布に合わせる'],
    ['C', 'CFair', '全体共通の class weight（balanced error）と、Y ごとに別の adversary'],
  ];
  s.addTable([
    [{ text: '', options: H }, { text: '手法', options: H }, { text: '根拠', options: H }],
    ...rows.map((r, i) => r.map(t => ({ text: t, options: i % 2 ? { fill: { color: THEME_TINT } } : {} }))),
  ], {
    x: 0.5, y: 1.1, w: 9, colW: [0.4, 3.0, 5.6], fontFace: F, fontSize: 10.5, color: INK,
    border: { type: 'solid', pt: 0.5, color: RULE }, valign: 'middle', margin: [3, 5, 3, 5],
  });
}

// 4-5. GroupDRO（左に説明、右に模式図）
const FIG = process.argv[3] || path.join(__dirname, 'figures');
{
  const s = slide('2. GroupDRO の問題 ①：global な class weight を一律に適用');
  bullets(s, [
    ['サブグループごとに陽性率が明確に異なる'],
    ['陽性率の高いサブグループほど、高いクラス重みが多くのサンプルに割り当てられ、損失が大きくなる'],
    ['右図：どのサンプルも損失 ℓ が同じ（識別の難しさが同じ）でも、群 X の損失は群 Y の約 2 倍', 1],
    ['GroupDRO の q は群 X に寄る'],
    ['識別の難しさ・バイアスではなく、陽性率の違いに大きく依存した学習調整になる', 1],
  ], 1.1, 3.9, 4.3);
  s.addImage({ path: FIG + '/global-class-weight.png', x: 5.0, y: 1.1, w: 4.5, h: 4.5 * 3.4 / 4.4 });
}
{
  const s = slide('2. GroupDRO の問題 ②：群ごとに class weight を用意');
  bullets(s, [
    ['サブグループごとに最適化している目的関数が異なる'],
    ['重み付き BCE の最適な logit は、群ごとに log w+ だけずれる', 1],
    ['右図：陽性率 10% の群 Y は w+ = 9 となり、logit が約 2.2 右にずれる', 1],
    ['群内の順位は保たれるが、全群をまとめるとスコア分布が揃わない'],
    ['共通のしきい値では群 Y の陰性が陽性側に入る', 1],
    ['global な識別性能を大きく低下させる', 1],
  ], 1.1, 3.9, 4.3);
  s.addImage({ path: FIG + '/per-group-class-weight.png', x: 5.0, y: 1.1, w: 4.5, h: 4.5 * 3.4 / 4.4 });
}

// 5. Proposal
{
  const s = slide('解決案と論点');
  bullets(s, [
    ['解決案：「群の難しさはサブグループごとの class weight で測る」＋「最適化は global な class weight で行う」'],
    ['上記の ① と ② の問題はどちらも解決できそう', 1],
    ['論点：「Hypernetwork に公平性損失を割り当てる」という主目的に対して、class weight を考慮した試行錯誤を行うべきかは不明'],
  ]);
}

pres.writeFile({ fileName: OUT });
