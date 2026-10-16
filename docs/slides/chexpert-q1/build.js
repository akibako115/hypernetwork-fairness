// 研究進捗報告スライド slide-chexpert-q1.pptx を生成する（報告の全体像と Q1〜Q3）。
// 先に figures.py で figures/ に図を書き出してから実行する:
//   uv run python docs/slides/chexpert-q1/figures.py
//   NODE_PATH=<pptxgenjs のある node_modules> node docs/slides/chexpert-q1/build.js [出力 pptx] [アウトライン md]
// 数値は analysis/archive/latent_bias_hypernet/results/q1_definitions.csv から読む。植え込み実験の数値は
// analysis/archive/latent_bias_hypernet/reports/latent_subgroups.md と results/audit/a_pacersub_s42/planted.csv から写している。
// Q2 の数値は analysis/archive/latent_bias_hypernet/results/q2_*.csv、Q3 は analysis/dataset_gain_sources/results/*.csv から読む。
// 付録 A4（ResNet の予備実験）の数値は analysis/archive/conditional_fairness_one_stage/README.md から写している。
// 構成：本編（問いの分解 → Q1・Q2・Q3 ごとに「設定 → 結果 → まとめ」→ 全体のまとめ）、付録 A（補足の結果）。
const fs = require('fs');
const path = require('path');
const pptxgen = require('pptxgenjs');
const pres = new pptxgen();
pres.layout = 'LAYOUT_16x9'; // 10 x 5.625
const F = 'Yu Gothic';
const INK = '222222', MUTED = '666666', RULE = 'BBBBBB';
const THEME = '1F3B57', THEME_TINT = 'E8EEF3', ACCENT = 'C0504D', ACCENT_TINT = 'F6E3E2', GREY = '8C99A6';
const DECK = 'CheXpert を主軸にした検証の組み直し：研究進捗';
const OUT = process.argv[2] || path.join(__dirname, 'slide-chexpert-q1.pptx');
const OUTLINE = process.argv[3];
const FIG = path.join(__dirname, 'figures');
const outline = [];

let page = 0;
function frame(s) {
  s.background = { color: 'FFFFFF' }; page++;
  s.addShape(pres.shapes.LINE, { x: 0.5, y: 5.2, w: 9, h: 0, line: { color: RULE, width: 0.5 } });
  s.addText(DECK, { x: 0.5, y: 5.25, w: 7, h: 0.25, fontFace: F, fontSize: 8, color: MUTED, margin: 0 });
  s.addText(String(page), { x: 9.0, y: 5.25, w: 0.5, h: 0.25, fontFace: F, fontSize: 8, color: MUTED, align: 'right', margin: 0 });
}
// crumb はピラミッドの中の位置（右肩に出す）
function slide(title, crumb, message) {
  const s = pres.addSlide(); frame(s);
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.7, fill: { color: THEME }, line: { color: THEME } });
  s.addText(title, { x: 0.5, y: 0, w: 6.6, h: 0.7, fontFace: F, fontSize: 18, bold: true, color: 'FFFFFF', valign: 'middle', margin: 0 });
  if (crumb) s.addText(crumb, { x: 7.1, y: 0, w: 2.4, h: 0.7, fontFace: F, fontSize: 9, color: 'C9D3DC', align: 'right', valign: 'middle', margin: 0 });
  s.addText(message, { x: 0.5, y: 0.8, w: 9, h: 0.62, fontFace: F, fontSize: 14, bold: true, color: THEME, valign: 'middle', margin: 0 });
  s.addShape(pres.shapes.LINE, { x: 0.5, y: 1.48, w: 9, h: 0, line: { color: THEME, width: 0.75 } });
  outline.push(`## ${page}. ${title}\n\n**${message}**\n`);
  return s;
}
// items: [text, level]。level 1 は下位の箇条
function bullets(s, items, { x = 0.5, y = 1.6, w = 4.45, h = 3.2, size = 11.5 } = {}) {
  const runs = items.map(([t, lv = 0], i) => ({
    text: t,
    options: { bullet: { indent: 12 }, indentLevel: lv, fontSize: lv ? size - 1 : size, color: lv ? '444444' : INK,
      paraSpaceAfter: 5, breakLine: i < items.length - 1 },
  }));
  s.addText(runs, { x, y, w, h, fontFace: F, valign: 'top', margin: 0 });
  outline.push(items.map(([t, lv = 0]) => `${'  '.repeat(lv)}- ${t}`).join('\n') + '\n');
}
function figure(s, name, ratio = 4.6 / 3.3, { x = 5.1, y = 1.6, w = 4.4 } = {}) {
  let h = w / ratio;
  if (h > 3.25) { h = 3.25; w = h * ratio; x = 9.5 - w; }
  s.addImage({ path: path.join(FIG, `${name}.png`), x, y, w, h });
}
function refs(s, text) {
  s.addText(text, { x: 0.5, y: 4.88, w: 9, h: 0.28, fontFace: F, fontSize: 7.5, color: MUTED, valign: 'bottom', margin: 0 });
  outline.push(`出典：${text}\n`);
}
function box(s, text, o) {
  s.addShape(pres.shapes.RECTANGLE, { x: o.x, y: o.y, w: o.w, h: o.h, fill: { color: o.fill || THEME_TINT },
    line: { color: o.line || o.fill || THEME_TINT, width: 0.75 } });
  s.addText(text, { x: o.x + 0.06, y: o.y + 0.03, w: o.w - 0.12, h: o.h - 0.06, fontFace: F, fontSize: o.size || 9,
    color: o.color || INK, bold: !!o.bold, align: o.align || 'center', valign: o.valign || 'middle', margin: 0 });
}
function link(s, x1, y1, x2, y2) {
  s.addShape(pres.shapes.LINE, { x: Math.min(x1, x2), y: y1, w: Math.abs(x2 - x1), h: y2 - y1,
    flipH: x2 < x1, line: { color: GREY, width: 0.75 } });
}
// 模式図の矢印（任意の向き）
function arrow(s, x1, y1, x2, y2) {
  s.addShape(pres.shapes.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1),
    flipH: x2 < x1, flipV: y2 < y1, line: { color: '555555', width: 1, endArrowType: 'triangle' } });
}
// 模式図の箱（白地・枠つき）
function node(s, text, x, y, w, h, fill = 'FFFFFF', line = GREY, size = 8.5) {
  box(s, text, { x, y, w, h, fill, line, size, color: INK });
}
// 2 列の表（左が項目名）
function kv(s, rows, { x = 0.5, y = 2.15, w = 5.25, labelW = 1.2, size = 9 } = {}) {
  s.addTable(rows.map(([k, v], i) => [
    { text: k, options: { bold: true, color: THEME, fill: { color: THEME_TINT } } },
    { text: v, options: i % 2 ? {} : {} },
  ]), { x, y, w, colW: [labelW, w - labelW], fontFace: F, fontSize: size, color: INK,
    border: { type: 'solid', pt: 0.5, color: RULE }, valign: 'middle', margin: [3, 5, 3, 5] });
  outline.push(rows.map(([k, v]) => `- ${k}：${v}`).join('\n') + '\n');
}
function intent(s, text, { w = 9 } = {}) {
  s.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 1.58, w, h: 0.48, fill: { color: THEME_TINT }, line: { color: THEME_TINT } });
  s.addText([
    { text: '意図　', options: { bold: true, color: THEME } },
    { text, options: { color: INK } },
  ], { x: 0.6, y: 1.58, w: w - 0.2, h: 0.48, fontFace: F, fontSize: 9.5, valign: 'middle', margin: 0 });
  outline.push(`意図：${text}\n`);
}
function table(s, header, rows, colW, { y = 1.6, size = 8.5, hi = [] } = {}) {
  const H = { bold: true, color: 'FFFFFF', fill: { color: THEME } };
  s.addTable([
    header.map((t) => ({ text: t, options: H })),
    ...rows.map((r, i) => r.map((t) => ({ text: t, options: { ...(i % 2 ? { fill: { color: THEME_TINT } } : {}), ...(hi.includes(i) ? { bold: true, color: ACCENT } : {}) } }))),
  ], { x: 0.5, y, w: 9, colW, fontFace: F, fontSize: size, color: INK, border: { type: 'solid', pt: 0.5, color: RULE }, valign: 'middle', margin: [2, 4, 2, 4] });
  outline.push(rows.map((r) => `- ${r.join(' / ')}`).join('\n') + '\n');
}


// q1_definitions.csv を読む（値にカンマを含まない前提）
const CSV = path.join(__dirname, '../../../analysis/archive/latent_bias_hypernet/results/q1_definitions.csv');
const [head, ...lines] = fs.readFileSync(CSV, 'utf8').trim().split('\n');
const cols = head.split(',');
const Q1 = lines.map((l) => Object.fromEntries(l.split(',').map((v, i) => [cols[i], v])));
function q1(dataset, definition) {
  const r = Q1.find((x) => x.dataset === dataset && x.definition === definition);
  if (!r) throw new Error(`${dataset} / ${definition} が q1_definitions.csv に無い`);
  return r;
}
const f2 = (v) => Number(v).toFixed(2);
const ci = (r, k) => `${f2(r[k])} [${f2(r[`${k}_lo`])}, ${f2(r[`${k}_hi`])}]`;

// 結果の CSV を読む（値にカンマを含まない前提）
function readCsv(rel) {
  const [h, ...ls] = fs.readFileSync(path.join(__dirname, rel), 'utf8').trim().split('\n');
  const c = h.split(',');
  return ls.map((l) => Object.fromEntries(l.split(',').map((v, i) => [c[i], v])));
}
const Q2 = readCsv('../../../analysis/archive/latent_bias_hypernet/results/q2_fairness_loss.csv');
const Q2T = readCsv('../../../analysis/archive/latent_bias_hypernet/results/q2_threshold_equivalence.csv');
const Q2D = readCsv('../../../analysis/archive/latent_bias_hypernet/results/q2_auroc_decomposition.csv');
const Q2W = readCsv('../../../analysis/archive/latent_bias_hypernet/results/q2_fairness_weights.csv');
const GAINS = readCsv('../../../analysis/dataset_gain_sources/results/gains.csv');
const INFO = readCsv('../../../analysis/dataset_gain_sources/results/attribute_information.csv');
const PAIRS = readCsv('../../../analysis/dataset_gain_sources/results/ham_pairs.csv');
function find(rows, pred, what) {
  const r = rows.find(pred);
  if (!r) throw new Error(`${what} が結果の CSV に無い`);
  return r;
}
const f3 = (v) => Number(v).toFixed(3);
const sg = (v, d = 3) => (Number(v) >= 0 ? '+' : '−') + Math.abs(Number(v)).toFixed(d);
const dci = (r, k, d = 3) => `${sg(r[`${k}_diff`], d)} [${sg(r[`${k}_diff_lo`], d)}, ${sg(r[`${k}_diff_hi`], d)}]`;
const gain = (ds, c) => find(GAINS, (x) => x.dataset === ds && x.split === 'test' && x.contrast === c, `${ds}/${c}`);
const gci = (r) => `${sg(r.delta)} [${sg(r.lo)}, ${sg(r.hi)}]`;
const info = (ds, q) => Number(find(INFO, (x) => x.dataset === ds && x.quantity === q, `${ds}/${q}`).value);
const PROC = { onestage: '1 段', insample: '2 段 in-sample', crossfit: '2 段 cross-fit' };
const LOSS = { eo: 'EO', minimax_auc: 'minimax AUC' };

// 1. 表紙
{
  const s = pres.addSlide(); frame(s);
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 1.7, w: 10, h: 1.8, fill: { color: THEME }, line: { color: THEME } });
  s.addText('属性の偏りを見つけ、直せるか', { x: 0.5, y: 1.85, w: 9, h: 0.75, fontFace: F, fontSize: 24, bold: true, color: 'FFFFFF', valign: 'middle', margin: 0 });
  s.addText('CheXpert を主軸にした検証の組み直しと、Q1〜Q3 の結果', { x: 0.5, y: 2.6, w: 9, h: 0.4, fontFace: F, fontSize: 14, color: 'D6E0E8', valign: 'middle', margin: 0 });
  s.addText('研究進捗報告　2026-10-16', { x: 0.5, y: 3.0, w: 9, h: 0.35, fontFace: F, fontSize: 12, color: 'D6E0E8', valign: 'middle', margin: 0 });
  outline.push('# 属性の偏りを見つけ、直せるか\n');
}

// 2. 報告の全体像
{
  const s = slide('報告の全体像', '全体像', 'CheXpert を主軸に、問いを「見つける」「直す」「効く条件」の 3 つに分けて組み直した');
  table(s, ['問い', '中身', '主軸', '補足'], [
    ['Q1 発見', '部分群の切り方で、見える性能差が変わるか', 'CheXpert', 'ISIC'],
    ['Q2 是正', '条件・損失・学習の段のどれを変えれば、属性の偏りが縮むか', 'CheXpert', 'HAM10000・FairVLMed'],
    ['Q3 条件', 'データをどう操作すれば、振り分けが効くようになるか', 'CheXpert（元データと並べる）', 'HAM10000'],
  ], [1.2, 4.6, 1.8, 1.4], { size: 11 });
  bullets(s, [
    ['Q1：属性を基準に、画像の特徴から作った部分群と、撮影条件などの交絡の候補を、同じ指標で比べた'],
    ['Q2：属性で条件付けた hypernet に公平性損失をかけ、学習の手順 3 つで比べた'],
    ['Q3：条件付けが効いた HAM10000 と、効かなかった CheXpert・FairVLMed を同じ物差しで比べた（手がかり）'],
  ], { y: 3.2, w: 9, h: 1.4, size: 12 });
}

// 3. Q1 比べたもの
{
  const s = slide('Q1：比べた部分群の定義', 'Q1 発見', '属性を基準に、画像特徴のクラスタと交絡の候補を、群の数をそろえて同じ指標で比べた');
  table(s, ['区分', '定義', '群の数'], [
    ['基準', '属性：年齢（65 歳で 2 区分）× race', '8'],
    ['画像特徴のクラスタ', 'k-means・GMM（LHCF と同じ）：基盤モデルの埋め込み（RAD-DINO・DINOv2）\nGEORGE：ERM モデル自身の特徴を、陽性・陰性それぞれの中で GMM（原典どおり）', '8（基準とそろえる）'],
    ['交絡の候補：撮影文脈', '撮影方向（正面 PA・正面 AP・側面）× 追跡撮影か × レポート種別', '18'],
    ['交絡の候補：画像に写る部分集団', 'ペースメーカー・PICC・中心静脈ライン・胸骨切開・手術クリップ', '各 2（有無）'],
  ], [2.2, 5.3, 1.5], { size: 10.5 });
  bullets(s, [
    ['クラスタは 1 段目が見ていない train_b に当て、test に割り当てた'],
    ['ERM の特徴を k-means・GMM で切ると、スコアの高さで群が分かれて群内の AUROC が機械的に下がるので使わない。GEORGE はクラスの中で切るので、原典どおり ERM の特徴を使う'],
    ['GEORGE は群が片方のクラスだけなので、群の AUROC は「陽性の群 vs 全陰性」で求める（他と同じ物差しではない）'],
  ], { y: 3.75, w: 9, h: 1.0, size: 10.5 });
  refs(s, 'LHCF：Masroor et al., MICCAI 2026　GEORGE：Sohoni et al., NeurIPS 2020');
}

// 4. Q1 設定
{
  const s = slide('Q1：モデルと指標', 'Q1 発見', '医用画像の公平性で標準の 4 指標を使い、属性との差を患者単位の bootstrap で比べた');
  kv(s, [
    ['モデル', 'ImageNet 初期化の ResNet-50 を ERM で学習（No Finding の 2 値）。train を患者で二分した片方だけで学習。seed 42 の 1 本'],
    ['データ', 'CheXpert の test 19,617 枚（陽性 10%）。補足に ISIC 2019 の施設間（BCN20000 で学習 → HAM10000 で評価）'],
    ['指標', '順位：最悪群 AUROC、AUROC gap（最大 − 最小）\nしきい値：Eopp1（TPR の最大 − 最小）、Eopp0（FPR の最大 − 最小）。しきい値は val の Youden'],
    ['比べ方', '各定義の値から属性の値を引いた差。95% 区間は患者単位の bootstrap（1,000 回）で、同じ標本の上で差を取る'],
    ['群の下限', '陽性・陰性とも 10 件以上の群だけを評価する'],
  ], { y: 1.65, w: 9, labelW: 1.4, size: 10.5 });
}

// 5. Q1 結果
{
  const s = slide('Q1 結果：定義ごとの性能差（CheXpert）', 'Q1 発見', '基盤モデルのクラスタでは順位の差は属性と変わらず、FPR の差だけが大きい。ERM の特徴で切る GEORGE では、TPR・FPR の差がともに大きい');
  s.addImage({ path: path.join(FIG, 'q1_values_chexpert.png'), x: 0.8, y: 1.58, w: 8.4, h: 8.4 * 844 / 2232 });
  const base = q1('chexpert', '年齢 × race');
  refs(s, `GEORGE は群が片方のクラスだけなので群の AUROC は定義できない。TPR 差は陽性の 4 群、FPR 差は陰性の 4 群で求めた。属性の値：最悪群 AUROC ${ci(base, 'worst_auroc')}、AUROC gap ${ci(base, 'auroc_gap')}、Eopp1 ${ci(base, 'eopp1')}、Eopp0 ${ci(base, 'eopp0')}`);
  outline.push('（図：定義ごとの値と 95% 区間、属性の値に縦線、4 指標）\n');
}

// 6. Q1 結果の読み
{
  const s = slide('Q1 結果の読み', 'Q1 発見', 'FPR の差は、画像の種類で群が分かれ、群ごとに陰性のスコア分布が違うことから来ていると読める');
  const k = q1('chexpert', 'k-means（raddino）'), c = q1('chexpert', '撮影文脈'), a = q1('chexpert', '年齢 × race');
  bullets(s, [
    ['基盤モデルのクラスタ：FPR の差（Eopp0）が属性より +0.17〜+0.41 大きい。撮影文脈は +0.47'],
    [`群の有病率の幅は、属性 ${f2(a.prev_min)}〜${f2(a.prev_max)} に対し、k-means（RAD-DINO）${f2(k.prev_min)}〜${f2(k.prev_max)}、撮影文脈 ${f2(c.prev_min)}〜${f2(c.prev_max)}`, 1],
    ['陰性のスコア分布が群で違うと、共通のしきい値では FPR が群で変わる。ここは解釈で、まだ分解はしていない', 1],
    ['順位（最悪群 AUROC・AUROC gap）は、どの定義も属性と区別できない'],
    ['GEORGE（ERM の特徴、クラスの中で切る）：TPR 差 0.92、FPR 差 0.68 と、属性（0.42、0.21）よりはるかに大きい'],
    ['モデル自身がスコアを低く付けた陽性が 1 つの群に集まる。「部分集団」より「モデルにとって難しい例」を切り出していると読める', 1],
    ['ISIC（補足、付録）でも、基盤モデルのクラスタは同じ傾向（k-means で Eopp0 が +0.17）。GEORGE の大きな差は ISIC では出ない'],
  ], { w: 9, size: 12 });
}

// 7. 交絡の候補
{
  const s = slide('Q1：交絡の候補で切ると', 'Q1 発見', '撮影文脈で切ると FPR の差が大きい。画像に写る部分集団は、単独では差が小さい');
  const rows = [['撮影文脈（18 群）', '撮影文脈']].concat(
    [['ペースメーカー', 'pacer'], ['PICC', 'picc'], ['中心静脈ライン', 'central_line'], ['胸骨切開', 'sternotomy'], ['手術クリップ', 'surgical_clip']]
      .map(([ja, key]) => [ja, `部分集団：${key}`]),
  ).map(([label, key]) => {
    const r = q1('chexpert', key);
    return [label, `${r.min_n}（陽性 ${r.min_pos}）`, ci(r, 'worst_auroc'), ci(r, 'eopp1'), ci(r, 'eopp0')];
  });
  const a = q1('chexpert', '年齢 × race');
  rows.unshift(['（基準）年齢 × race', `${a.min_n}（陽性 ${a.min_pos}）`, ci(a, 'worst_auroc'), ci(a, 'eopp1'), ci(a, 'eopp0')]);
  table(s, ['定義', '最小群の枚数', '最悪群 AUROC', 'Eopp1（TPR 差）', 'Eopp0（FPR 差）'], rows, [2.0, 1.5, 1.85, 1.85, 1.8], { size: 9.5, hi: [1] });
  bullets(s, [
    ['部分集団は 2 群なので、8 群の属性と値をそのまま比べることはできない。目立つのは胸骨切開の TPR 差（陽性 92 件）'],
    ['これらが属性間の差をどれだけ作っているか（構成をそろえた比較）は、まだ測っていない'],
  ], { y: 4.05, w: 9, h: 0.8, size: 10 });
}

// 8. 状況による違い
{
  const s = slide('Q1：どういう偏りなら、どの切り方で見えるか', 'Q1 発見', '属性に沿わない偏りは属性で切ると消え、クラスタでも群が粗いと見えない（旧設定の植え込み実験）');
  intent(s, '正解の分かっている偏りを学習データに植え込み、各切り方で見えるかを確かめた。指標・群の数が上の設定とそろっていない旧設定の結果');
  table(s, ['状況', '植え込み', '属性で切る', '画像特徴のクラスタで切る'], [
    ['属性に沿わない部分集団を学び損ねる', 'ペースメーカーに触れる陽性を 10% に間引く（その部分集団の log O/E +0.62、TPR 0.49）', '見えない。ペースメーカーが最も多いセルでも割合 20%、log O/E −0.08', '32 群なら見える（ペースメーカー 81% のクラスタ、log O/E +0.57）。8 群では割合 18% で見えない'],
    ['属性と画像要因の交差に偏りがある', '65 歳未満の女性 × デバイスの陽性を陰性に反転', '見える。モデルが偏りを属性の軸に沿って広げた（デバイスの無い若年女性にも log O/E +0.27）', '—'],
  ], [2.0, 2.4, 2.3, 2.3], { y: 2.2, size: 9 });
  bullets(s, [
    ['どの切り方が役に立つかは、偏りが属性の軸に沿うかと、群の細かさの両方で決まる'],
    ['log O/E：群の観測陽性数 ÷ 予測確率の和（の対数）。正なら過少予測', 1],
  ], { y: 4.0, w: 9, h: 0.85, size: 10 });
}

// 9. まとめと次の課題
{
  const s = slide('Q1 のまとめと次の課題', 'Q1 発見', '基盤モデルのクラスタでは FPR の差だけが変わり、ERM の特徴で切るとモデルにとって難しい例が集まって、TPR・FPR の差が大きく見えた');
  bullets(s, [
    ['分かったこと'],
    ['順位（AUROC）の差は、属性・基盤モデルのクラスタ・撮影文脈のどれで切っても同程度', 1],
    ['基盤モデルのクラスタと撮影文脈で切ると、FPR の差が属性より大きく見える', 1],
    ['ERM の特徴をクラスの中で切る（GEORGE）と、モデルにとって難しい例が集まり、TPR・FPR の差が大きく見える', 1],
    ['属性に沿わない偏りは、属性で切ると見えない（植え込み）', 1],
    ['次の課題'],
    ['FPR の差が、較正のずれによるのか、群の症例の構成によるのかを分ける', 1],
    ['撮影文脈や部分集団が、属性間の差をどれだけ作っているかを測る（構成をそろえた比較）', 1],
    ['ERM の特徴からスコアの方向を除いてクラスタにする。hypernet が生成する補正で部分群を定義する', 1],
    ['植え込み実験を今の設定で回し直す。1 段目の seed を増やす', 1],
  ], { w: 9, size: 12.5 });
}


// 10. Q2 問いと設定
{
  const s = slide('Q2：問いと設定', 'Q2 是正', '属性で条件付けた hypernet に公平性損失をかけ、学習の手順 3 つで、群の性能差が縮むかを比べた');
  kv(s, [
    ['問い', '属性で条件付けた hypernet は、ERM のままで、または公平性損失をかけたときに、群の性能差を縮めるか。手順で結論が変わるか'],
    ['モデル', '属性（年齢・性別・race・ethnicity・撮影方向）で条件付けた Spatial LoRA'],
    ['群・指標', '年齢（65 歳で 2 区分）× race の 8 群。指標は Q1 と同じ 4 つと全体 AUROC。3 seed の平均、患者単位の bootstrap 95%'],
    ['損失', 'ERM・EO rate（Cotter et al. 2019）・minimax AUC（Yang et al. 2023）。群の重みは学習に使わない held-out（train の患者の 20%）で更新'],
    ['手順', '1 段：ImageNet から全層を学習（12.6 万枚）\n2 段 in-sample：train 全体で学習した ResNet-50 を凍結し、1 段と同じデータで adapter を学習\n2 段 cross-fit：train の半分で学習した ResNet-50 を凍結し、残り半分で adapter を学習（6.3 万枚）'],
    ['比べ方', '手順の中での「損失 − ERM」だけを読む（手順の間でデータの量が違う）。epoch は val の全体 AUROC で選ぶ'],
  ], { y: 1.6, w: 9, labelW: 1.2, size: 9.5 });
  refs(s, 'EO の task 重みの初期値だけ 0.9 に変えた（一様な初期値では全予測が片側に寄って崩れた）。GroupDRO は q が最悪群に向かず、代表にしない');
}

// 11. C1
{
  const s = slide('Q2 結果 1：ERM の hypernet', 'Q2 是正', 'ERM で学習した hypernet は、ResNet-50 の ERM と変わらない。例外の HAM10000 も、改善は全体性能で公平性ではない');
  table(s, ['データ', '属性条件付き Spatial LoRA − ResNet', 'HyperAdapt − ResNet', '群間の差'], [
    ['CheXpert', gci(gain('chexpert', 'meta − resnet')), gci(gain('chexpert', 'hyperadapt − resnet')), '最悪群 AUROC・gap とも雑音の範囲'],
    ['FairVLMed', gci(gain('fairvlmed', 'meta − resnet')), gci(gain('fairvlmed', 'hyperadapt − resnet')), 'race 軸の gap・最悪群に差なし'],
    ['HAM10000', gci(gain('ham10000', 'meta − resnet')), gci(gain('ham10000', 'hyperadapt − resnet')), '性別 × 年齢の gap・Eopp は変わらない'],
  ], [1.4, 2.8, 2.4, 2.4], { y: 1.65, size: 10, hi: [2] });
  bullets(s, [
    ['全体 test AUROC の差、3 seed の平均、患者（HAM は病変）単位の bootstrap 95%'],
    ['HAM10000 で上がる理由は Q3 で扱う'],
  ], { y: 3.3, w: 9, h: 1.2, size: 11 });
}

// 12. C2 損失の効果
{
  const s = slide('Q2 結果 2：公平性損失が変えるもの', 'Q2 是正', '損失は群内の順位の差（最悪群 AUROC・gap）を縮めず、動作点（Eopp1・Eopp0）だけを下げる。3 手順とも同じ');
  s.addImage({ path: path.join(FIG, 'q2_loss_effects.png'), x: 0.5, y: 1.58, w: 9, h: 9 * 742 / 2481 });
  const w = Q2W.filter((x) => x.procedure !== 'onestage' && x.loss === 'minimax_auc').map((x) => Number(x.q_on_worst_pair));
  bullets(s, [
    [`2 段では群の重みが held-out で最悪の群に向いていた（minimax AUC の q は最悪ペアに ${Math.min(...w).toFixed(2)}〜${Math.max(...w).toFixed(2)}）。損失は意図どおり働いたうえでの結果で、実装や最適化の失敗とは言いにくい`],
    ['全体 AUROC は minimax AUC で −0.013〜−0.024 下がる', 0],
  ], { y: 4.35, w: 9, h: 0.5, size: 9.5 });
}

// 13. C2 しきい値
{
  const s = slide('Q2 結果 3：群別しきい値との比較', 'Q2 是正', '損失の動作点の改善は、ERM に val で決めた群別しきい値を当てれば同等以上になり、全体 AUROC も下がらない');
  s.addImage({ path: path.join(FIG, 'q2_threshold.png'), x: 0.5, y: 1.55, w: 9, h: 9 * 757 / 2474 });
  bullets(s, [
    ['群別しきい値：群ごとに、全体の動作点（全体 Youden のしきい値での TPR・FPR）に最も近い点を val で選ぶ。スコアは ERM のまま'],
    ['minimax AUC のスコアは「ERM のスコアの定数倍 + 群ごとの定数」でほぼ近似でき（2 段で val の R² 0.93〜0.95）、test の Eopp も再現する。群別しきい値をスコアに焼き込んだものに当たる'],
    ['例外：1 段の Eopp0 だけは損失のほうが低い（0.10〜0.11 対 0.18）。Eopp0 を直接狙うしきい値は未確認'],
  ], { y: 4.32, w: 9, h: 0.55, size: 8.5 });
}

// 14. C2 なぜ全体 AUROC が下がるか
{
  const s = slide('Q2 結果 4：全体 AUROC が下がる理由', 'Q2 是正', '1 つのスコアで群の動作点をそろえると、群をまたぐ順位が崩れて全体 AUROC が下がる。低下の 85〜95% がこの分');
  figure(s, 'q2_decomposition', 1531 / 662, { x: 5.0, y: 1.65, w: 4.5 });
  bullets(s, [
    ['全体 AUROC は、陽性・陰性の全組の一致率。同じ群の中の組（22%）と、群をまたぐ組（78%）に分けられる'],
    ['群ごとにスコアを平行移動させても、群の中の組は動かず、群をまたぐ組だけが動く', 1],
    ['陽性率が群で違うと、共通のしきい値で動作点をそろえるには、群のスコアの位置関係を崩すしかない'],
    ['残りは人数の多い群の群内 AUROC がわずかに下がった分（minimax AUC で −0.004〜−0.013）。悪い群は上がらない'],
    ['順位付けは ERM のスコア、判定は群別しきい値、と分ければ両立する。hypernet は属性を入力に取るので、群別しきい値を使えない理由も無い'],
  ], { w: 4.4, size: 10 });
}

// 15. Q2 まとめ
{
  const s = slide('Q2 のまとめと次の課題', 'Q2 是正', '公平性のために hypernet を使うメリットは見つからない。損失の効果は、後から群別しきい値を当てることで代わりが利く');
  bullets(s, [
    ['分かったこと'],
    ['ERM の hypernet は ResNet-50 と変わらない（CheXpert・FairVLMed）', 1],
    ['公平性損失は群内の順位の差を縮めず、動作点だけを変える。1 段・2 段のどの手順でも同じ', 1],
    ['動作点の改善は、ERM + 群別しきい値で同等以上。全体 AUROC の低下は、しきい値をスコアに焼き込む代償', 1],
    ['ResNet-50 に同じ損失をかけた予備実験（seed 42、年齢 2 群）でも振る舞いは同じ', 1],
    ['次の課題'],
    ['損失自身の狙う量を、学習データ・held-out・test の 3 か所で追い、効きを失う場所を切り分ける', 1],
    ['Eopp0 を直接狙うしきい値との比較。1 段の群の重みの診断（今は最後の 1 バッチの値で粗い）', 1],
    ['Q2 と同じ設定の ResNet-50 の run', 1],
  ], { w: 9, size: 12 });
}

// 16. Q3 3 データの比較
{
  const s = slide('Q3：条件付けが効くのはどんなデータか', 'Q3 条件（手がかり）', '条件付けで上がる幅は、属性が「画像のスコアに無いラベルの情報」を運ぶ度合いで決まる');
  figure(s, 'q3_datasets', 1781 / 707, { x: 0.5, y: 1.6, w: 5.6 });
  const late = (ds) => sg(info(ds, 'late_fusion_valfit_gain_resnet'));
  bullets(s, [
    [`ResNet の出力に属性を後から足したときの上がり幅：HAM ${late('ham10000')}、FairVLMed ${late('fairvlmed')}、CheXpert ${late('chexpert')}。条件付けの上がり方と同じ順`],
    [`属性だけの予測力は 3 データとも高い（HAM ${f2(info('ham10000', 'attr_only_test'))}、FairVLMed ${f2(info('fairvlmed', 'attr_only_test'))}、CheXpert ${f2(info('chexpert', 'attr_only_test'))}）。CheXpert では、その情報が画像のスコアにすでに入っている`],
    ['FairVLMed は上がる余地が標本誤差（0.005）と同程度で、構成そのものの損と打ち消し合う'],
    ['画像から作った条件（潜在クラスタ・ランダム区分）は効かない'],
  ], { x: 6.25, w: 3.25, size: 9.5 });
  refs(s, '値は test、3 seed の平均。区間は患者（病変）単位の bootstrap 95%。後から足す重みは val で学習。analysis/dataset_gain_sources');
}

// 17. Q3 HAM の機構
{
  const s = slide('Q3：HAM10000 で上がる 2 つの機構', 'Q3 条件（手がかり）', '年齢による事前確率の注入と、年齢を条件が受け持つことで画像側の判定が良くなる作用の 2 つ');
  const pv = (pair, c) => find(PAIRS, (x) => x.split === 'test' && x.pair === pair && x.contrast === c, `${pair}/${c}`);
  const pc = (pair, c) => { const r = pv(pair, c); return `${sg(r.delta)} [${sg(r.lo)}, ${sg(r.hi)}]`; };
  table(s, ['区別する組', 'ResNet', '条件を定数', '属性条件付き', '属性条件付き（test 時に属性を固定）', '固定 − 定数'], ['nv', 'bkl', 'bcc'].map((d) => {
    const pair = `mel vs ${d}`;
    return [`黒色腫 vs ${{ nv: '母斑', bkl: 'bkl', bcc: 'bcc' }[d]}`, f3(pv(pair, 'resnet').delta), f3(pv(pair, 'const').delta), f3(pv(pair, 'meta').delta), f3(pv(pair, 'meta_fixed').delta), pc(pair, 'meta_fixed − const')];
  }), [1.5, 0.9, 1.1, 1.2, 2.2, 2.1], { y: 1.6, size: 9.5 });
  bullets(s, [
    ['黒色腫 vs 母斑：test 時に渡す年齢による事前確率の注入（母斑は平均 45 歳、黒色腫 61 歳）。属性を固定すると消え、後から足すだけ・分類層だけの条件付けでほぼ再現する'],
    ['黒色腫 vs bkl・bcc：属性を入れて学習したこと自体で、画像側の判定が良くなる。良性内のスコアと年齢の相関は ResNet 0.15 → 0.04 に下がり、年齢を条件が受け持って画像側が「老いた皮膚らしさ」を悪性の代わりに使わなくなった、という解釈と整合する'],
    ['この 2 つめは val ではほとんど出ず、分布のずれた公式 test でだけ大きい。全体性能の改善で、性別 × 年齢の公平性は変わらない'],
  ], { y: 2.95, w: 9, h: 1.9, size: 10 });
}

// 18. 全体のまとめ
{
  const s = slide('全体のまとめ', 'まとめ', '公平性のために hypernet（属性の条件付け）を使うメリットは、現時点では見つからなかった');
  bullets(s, [
    ['Q1 発見：順位（AUROC）の差は部分群の切り方で区別できず、FPR の差だけが画像の種類で群を切ると大きく見える'],
    ['Q2 是正：hypernet は ERM で ResNet と変わらず、公平性損失は動作点だけを変える。その効果は ERM + 群別しきい値で代わりが利く'],
    ['Q3 条件：条件付けが効くのは、属性が画像に無い情報を運ぶときに限られ、その効果は公平性ではなく全体性能に出る'],
    ['次の方向'],
    ['画像に無い情報（レポート文・臨床情報など）を条件にする PoC。効くかは「出力に後から足したときの上がり幅」で学習前に見積もる', 1],
    ['先行研究（HyperAdapt・MoE 系）の設定での再現と、私たちの対照（定数条件・分類層だけ・入れ替え）の追加', 1],
    ['HAM で年齢だけで条件付ける・外す学習（肩代わりの因果の確認）', 1],
  ], { w: 9, size: 12.5 });
}

// 付録 A1. ISIC
{
  const s = slide('付録 A1：Q1 の結果（ISIC 2019、施設間）', '付録', 'ISIC でも AUROC の差は属性と区別できず、k-means で FPR の差が大きく見える');
  s.addImage({ path: path.join(FIG, 'q1_values_isic.png'), x: 0.8, y: 1.58, w: 8.4, h: 8.4 * 640 / 2232 });
  const b = q1('isic', '性別 × 年齢');
  bullets(s, [
    ['BCN20000 で学習した ResNet-50 を HAM10000 の test（4,031 枚、陽性 12%）で評価。基準は性別 × 年齢の 4 群、クラスタも 4 群'],
    [`属性の値：最悪群 AUROC ${ci(b, 'worst_auroc')}、Eopp1 ${ci(b, 'eopp1')}、Eopp0 ${ci(b, 'eopp0')}`, 1],
    ['GMM は有効な群が 3 つしか残らなかった。GEORGE の陰性の群は 2 つで、FPR がほぼ同じ', 1],
  ], { y: 4.1, w: 9, h: 0.75, size: 10 });
}


// 付録 A2. Q2 の全表
{
  const s = slide('付録 A2：Q2 の結果（全表）', '付録', '最悪群 AUROC・gap はどの手順・損失でも ERM と区別できず、Eopp1・Eopp0 だけが下がる');
  const rows = [];
  for (const p of ['onestage', 'insample', 'crossfit']) {
    const e = find(Q2, (x) => x.condition === `${p}:erm:auroc`, `${p} ERM`);
    rows.push([PROC[p], 'ERM（値）', f3(e.overall_auroc), f3(e.worst_auroc), f3(e.auroc_gap), f2(e.eopp1), f2(e.eopp0)]);
    for (const l of ['eo', 'minimax_auc']) {
      const r = find(Q2, (x) => x.condition === `${p}:${l}:auroc`, `${p} ${l}`);
      rows.push(['', `${LOSS[l]}（差）`, sg(Number(r.overall_auroc) - Number(e.overall_auroc)), dci(r, 'worst_auroc'), dci(r, 'auroc_gap'), dci(r, 'eopp1', 2), dci(r, 'eopp0', 2)]);
    }
  }
  table(s, ['手順', '損失', '全体 AUROC', '最悪群 AUROC', 'AUROC gap', 'Eopp1', 'Eopp0'], rows, [1.2, 1.15, 0.85, 1.5, 1.5, 1.4, 1.4], { y: 1.6, size: 8 });
  refs(s, '損失の行は同じ手順の ERM との差 [95% 区間]。epoch は val の全体 AUROC で選択。損失自身の狙いで epoch を選んでも結論は同じ（results/q2_fairness_loss.csv）');
}

// 付録 A3. 群の重み
{
  const s = slide('付録 A3：損失は意図どおり働いたか', '付録', '2 段では群の重みが held-out で最も悪い群に向いていた。1 段の診断は粗く、向いていなかった');
  const rows = [];
  for (const p of ['onestage', 'insample', 'crossfit']) {
    for (const l of ['eo', 'minimax_auc']) {
      const w = Q2W.filter((x) => x.procedure === p && x.loss === l);
      const cell = l === 'eo'
        ? `制約に載った重み ${w.map((x) => f2(x.constraint_mass)).join('・')}、うち最悪 TPR の群を上げる重み ${w.map((x) => Number(x.mass_lifting_worst_tpr).toFixed(3)).join('・')}`
        : `最悪ペアに載った q ${w.map((x) => Number(x.q_on_worst_pair).toFixed(2)).join('・')}、q 最大のペアが最悪ペアか ${w.map((x) => (x.q_argmax_is_worst_pair === 'True' ? 'はい' : 'いいえ')).join('・')}`;
      rows.push([PROC[p], LOSS[l], cell]);
    }
  }
  table(s, ['手順', '損失', '学習の終わりの群の重み（seed 42・43・44）'], rows, [1.6, 1.3, 6.1], { y: 1.6, size: 9 });
  refs(s, '1 段の値は epoch の最後の 1 バッチ（16 セル × 8 件）から測った粗い診断。2 段は epoch 中の held-out の率の平均');
}

// 付録 A4. ResNet の予備実験
{
  const s = slide('付録 A4：ResNet-50 に公平性損失をかけた予備実験', '付録', 'ResNet-50 と Spatial LoRA で、公平性損失の振る舞いは同じだった（seed 42、年齢 2 群）');
  table(s, ['条件', 'AUROC', 'Eopp1', 'Eodds'], [
    ['ERM ResNet-50', '0.868', '0.241', '0.187'], ['ERM Spatial LoRA', '0.869', '0.294', '0.215'],
    ['ResNet-50 minimax AUC', '0.851', '0.138', '0.088'], ['Spatial LoRA minimax AUC', '0.854', '0.180', '0.110'],
    ['ResNet-50 EO rate', '0.861', '0.156', '0.107'], ['Spatial LoRA EO rate', '0.862', '0.123', '0.074'],
  ], [4.5, 1.5, 1.5, 1.5], { y: 1.6, size: 10 });
  bullets(s, [
    ['val の値、val AUROC 最大の epoch。群の重みは train のバッチで更新（Q2 の held-out 方式の前の設定）'],
    ['どちらも ERM より AUROC が 0.007〜0.019 下がり、動作点の差は学習の序盤だけ縮む'],
  ], { y: 3.9, w: 9, h: 0.9, size: 10 });
  refs(s, 'analysis/archive/conditional_fairness_one_stage（2026-10-08）');
}

pres.writeFile({ fileName: OUT }).then(() => {
  console.log(`wrote ${OUT}`);
  if (OUTLINE) { fs.writeFileSync(OUTLINE, outline.join('\n')); console.log(`wrote ${OUTLINE}`); }
});
