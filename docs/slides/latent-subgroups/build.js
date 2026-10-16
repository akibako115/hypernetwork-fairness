// 研究進捗報告スライド slide-latent-subgroups.pptx を生成する。
// 先に figures.py で figures/ に図を書き出してから実行する:
//   uv run python docs/slides/latent-subgroups/figures.py
//   NODE_PATH=<pptxgenjs のある node_modules> node docs/slides/latent-subgroups/build.js [出力 pptx] [アウトライン md]
// アウトライン md（タイトル・メッセージ・本文）を指定すると、文章の検査用に書き出す。
// 構成：本編（問いの分解 → 実験ごとに「設定 → 結果」）、付録 A（補足の結果）、付録 B（先行手法の説明、1 手法 1 枚）。
const fs = require('fs');
const path = require('path');
const pptxgen = require('pptxgenjs');
const pres = new pptxgen();
pres.layout = 'LAYOUT_16x9'; // 10 x 5.625
const F = 'Yu Gothic';
const INK = '222222', MUTED = '666666', RULE = 'BBBBBB';
const THEME = '1F3B57', THEME_TINT = 'E8EEF3', ACCENT = 'C0504D', ACCENT_TINT = 'F6E3E2', GREY = '8C99A6';
const DECK = '潜在部分群による条件付けと公平性：研究進捗';
const OUT = process.argv[2] || path.join(__dirname, 'slide-latent-subgroups.pptx');
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

// 問いの分解（verdicts を渡すと、各葉の下に判定と結果を出す）
const TOP = '属性を使わず、潜在部分群で振り分ければ、属性の公平性まで改善できるか';
const MID = [
  { t: 'Q1 発見：偏りを潜在部分群で見つけられるか', x: 0.5, w: 1.7 },
  { t: 'Q2 是正：潜在部分群で直せるか', x: 2.3, w: 3.55 },
  { t: 'Q3 条件：振り分けはどんな場で効くか', x: 5.95, w: 3.55 },
];
const LEAVES = [
  { q: 0, h: '属性セルの監査', b: '属性セルでは見えない偏りを、潜在スライスで見つけられる', d: 'CheXpert・ISIC', p: 'p.7–10' },
  { q: 1, h: 'a 事後補正', b: '凍結モデルに潜在の振り分け補正を足せば直る', d: 'CheXpert・ISIC', p: 'p.11–12' },
  { q: 1, h: 'b E2E 条件付け', b: 'hypernet を潜在部分群で条件付ければ、属性条件と同等以上', d: 'HAM10000・FairVLMed', p: 'p.13–16' },
  { q: 1, h: 'c 公平性損失', b: '潜在の群で重み付ける損失（LHCF 型）で直る', d: 'HAM10000', p: 'p.17–21' },
  { q: 2, h: 'a 正の対照', b: '規則が逆になる部分群があれば、振り分けは効く', d: 'HAM10000＋植え込み', p: 'p.22–23' },
  { q: 2, h: 'b 1 段か 2 段か', b: '凍結した 1 段目の上でも効く', d: 'HAM10000＋植え込み', p: 'p.24–25' },
  { q: 2, h: 'c 区分の作り直し', b: '誤りから区分を作り直せば、境界に近づく', d: 'HAM10000＋植え込み', p: 'p.26–27' },
];
function pyramid(s, verdicts) {
  const top = { x: 2.0, y: 1.6, w: 6.0, h: 0.45 };
  box(s, TOP, { ...top, fill: THEME, color: 'FFFFFF', bold: true, size: 11 });
  const midY = 2.3, midH = 0.5, leafY = 3.05, leafH = verdicts ? 0.95 : 1.5;
  MID.forEach((m) => {
    link(s, top.x + top.w / 2, top.y + top.h, m.x + m.w / 2, midY);
    box(s, m.t, { x: m.x, y: midY, w: m.w, h: midH, bold: true, size: 9.5, color: THEME });
  });
  const counts = [0, 0, 0];
  LEAVES.forEach((l) => counts[l.q]++);
  const seen = [0, 0, 0];
  LEAVES.forEach((l, i) => {
    const m = MID[l.q], n = counts[l.q], gap = 0.1;
    const w = (m.w - gap * (n - 1)) / n, x = m.x + seen[l.q]++ * (w + gap);
    link(s, m.x + m.w / 2, midY + midH, x + w / 2, leafY);
    s.addShape(pres.shapes.RECTANGLE, { x, y: leafY, w, h: leafH, fill: { color: 'FFFFFF' }, line: { color: GREY, width: 0.75 } });
    const body = [
      { text: l.h, options: { bold: true, fontSize: 9, color: THEME, breakLine: true } },
      { text: l.b, options: { fontSize: 8.5, color: INK, breakLine: !verdicts } },
    ];
    if (!verdicts) body.push({ text: `${l.d}　${l.p}`, options: { fontSize: 7.5, color: MUTED } });
    s.addText(body, { x: x + 0.05, y: leafY + 0.04, w: w - 0.1, h: leafH - 0.08, fontFace: F, valign: 'top', margin: 0, paraSpaceAfter: 2 });
    if (verdicts) {
      const [label, text] = verdicts[i];
      const fill = label === '支持' ? THEME : label === '不支持' ? ACCENT : GREY;
      box(s, label, { x, y: leafY + leafH + 0.04, w, h: 0.24, fill, color: 'FFFFFF', bold: true, size: 9 });
      box(s, text, { x, y: leafY + leafH + 0.31, w, h: 0.5, fill: label === '不支持' ? ACCENT_TINT : THEME_TINT, size: 8, valign: 'top' });
    }
  });
  outline.push(`（図：問いの分解）${TOP} → ${MID.map((m) => m.t).join(' / ')}\n`);
}

// 1. 表紙
{
  const s = pres.addSlide(); frame(s);
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 1.7, w: 10, h: 1.8, fill: { color: THEME }, line: { color: THEME } });
  s.addText('潜在部分群で振り分けても、属性の公平性は上がらなかった', { x: 0.5, y: 1.85, w: 9, h: 0.75, fontFace: F, fontSize: 24, bold: true, color: 'FFFFFF', valign: 'middle', margin: 0 });
  s.addText('属性を使わない条件付け・補正の検証（CheXpert・ISIC・HAM10000・FairVLMed）', { x: 0.5, y: 2.6, w: 9, h: 0.4, fontFace: F, fontSize: 14, color: 'D6E0E8', valign: 'middle', margin: 0 });
  s.addText('研究進捗報告　2026-10-11 時点', { x: 0.5, y: 3.0, w: 9, h: 0.35, fontFace: F, fontSize: 12, color: 'D6E0E8', valign: 'middle', margin: 0 });
  outline.push('# 潜在部分群で振り分けても、属性の公平性は上がらなかった\n');
}

// 2. 今日の結論
{
  const s = slide('今日の結論', '', '潜在部分群は属性セルに見えない較正の偏りを見つけるが、是正には効かない。振り分けが効くのは部分群で判定規則が食い違う場だけで、実データではその食い違いが見つかっていない');
  bullets(s, [
    ['Q1 発見：見える偏りの種類が切り方で分かれる（CheXpert・ISIC）'],
    ['属性セルには順位・しきい値の偏り（最悪群 AUROC・TPR 差）が出て、較正のずれは出ない。潜在スライスはその逆', 1],
    ['Q2 是正：事後補正・E2E の条件付け・公平性損失のどれでも、属性の最悪群 AUROC は有意に上がらない'],
    ['条件付きモデルの利得は、属性のように画像にない情報から来る', 1],
    ['Q3 条件：規則が逆になる部分群を植え込むと、境界に合った区分の振り分けだけが少数側の AUROC を +0.094 上げる'],
    ['ただし、その境界をモデルの誤りから見つける方法はまだない', 1],
    ['相談したいこと：否定的な結果を軸にした論文の筋と、次に検証する場の選び方'],
  ], { w: 9, size: 14 });
}

// 3. 背景
{
  const s = slide('背景：属性の条件付けと、属性に沿わない偏り', '背景', '属性で条件付けるモデルは推論時に属性を要し、属性の軸に沿わない偏りには届かない');
  bullets(s, [
    ['属性（年齢・性別）で条件付けた hypernetwork や MoE が、医用画像の公平性を改善すると報告されている［1, 2］'],
    ['推論時にも属性が要る。HAM10000 の新施設のように属性が欠けると、性能が落ちる', 1],
    ['偏りは属性の軸に沿うとは限らない'],
    ['同じラベルの中に、性能の低い部分群が隠れている（hidden stratification）［3］', 1],
    ['胸部 X 線の性差の背後に、胸腔ドレーンや ECG ワイヤのショートカットがあった［4］', 1],
    ['属性を使わない流れもある。埋め込みのクラスタで隠れコホートを作り、最悪コホートの損失を下げる［5, 6］'],
    ['本研究の問い：属性の代わりに潜在空間の部分群で振り分ければ、属性を使わずに属性の公平性まで改善できるか'],
  ], { w: 9, size: 12.5 });
  refs(s, '［1］Xu et al., arXiv 2601.13094, 2026　［2］Park et al. (dMoE), arXiv 2502.00619, 2025　［3］Oakden-Rayner et al., CHIL 2020　［4］Olesen et al., FAIMI 2024　［5］Sohoni et al. (GEORGE), NeurIPS 2020　［6］Masroor et al. (LHCF), MICCAI 2026');
}

// 4. 問いの分解
{
  const s = slide('問いの分解', '全体像', '問いを「見つけられるか」「直せるか」「どんな場なら効くか」に分け、直し方は 3 通り試した');
  pyramid(s, null);
}

// 5. 共通の設定
{
  const s = slide('共通の設定：データ・モデル・標準の指標', '全体像', '4 つのデータセットで ResNet-50 系のモデルを 3 seed ずつ学習し、属性群の最悪 AUROC を主な物差しにした');
  table(s, ['データ', 'タスクと分割', '属性', '使った問い'], [
    ['CheXpert（胸部 X 線）', 'No Finding の 2 値。train を患者単位で A・B に二分', '性別・年齢・人種', 'Q1・Q2-a'],
    ['ISIC 2019（皮膚）', 'BCN20000 で学習し、HAM10000 で監査・評価。メラノーマの 2 値', '性別・年齢・部位', 'Q1・Q2-a'],
    ['HAM10000（皮膚）', '悪性（mel・akiec）の 2 値。train 8,457／val 1,491／test 1,511 枚（ISIC2018 公式 test。属性の欠けた新施設を含む）', '性別・年齢・部位', 'Q2-b・Q2-c・Q3'],
    ['FairVLMed（眼底 SLO）', '緑内障の 2 値。train 7,000／test 2,000 枚', '人種・性別ほか', 'Q2-b'],
  ], [1.7, 4.6, 1.4, 1.3], { size: 9 });
  bullets(s, [
    ['最悪群 AUROC：性別 × 年齢（65 歳以上か）の 4 群ごとに AUROC を出し、その最小値。医用画像の公平性研究で標準的な指標［1］。群を細かく切りすぎると標本誤差で比べられないので 4 群に絞った'],
    ['external：HAM10000 の test のうち、学習にない施設の画像。属性が欠けている'],
    ['差と区間：比べる相手との差の 3 seed 平均。95% 区間は、同じ test 画像の上で病変（患者）単位の bootstrap で求める。区間が 0 を含まなければ「有意」と呼ぶ'],
    ['潜在部分群：属性でなく、画像の埋め込みのクラスタで作る群（例：DINOv2 の埋め込みを GMM で 8 群に分ける）'],
  ], { y: 3.35, w: 9, h: 1.55, size: 9.5 });
  refs(s, '［1］Zong et al. (MEDFAIR), ICLR 2023');
}

// 6. 独自に足した指標
{
  const s = slide('独自に足した指標と、採用した理由', '全体像', '標準の AUROC だけでは「なぜ効かないか」を切り分けられないので、仕組みを直接見る指標を 4 つ足した');
  table(s, ['指標', '何を測るか', '採用した理由', '使った所'], [
    ['群ごとの log O/E', '群で観測した陽性数と、予測確率の和の比（の対数）。予測の水準のずれ', 'AUROC は群内の順位しか見ず、群ごとの過少・過大予測を見落とす。後から足す補正が直すのはこのずれなので、直接測る（次々ページ）', 'Q1・Q2-a'],
    ['条件の入れ替えによる AUROC の低下', 'test の画像はそのままで、条件だけを画像の間で入れ替えたときの低下', '性能が同じでも、「条件を使っていない」のか「使ったが効かない」のかは区別できない。入れ替えて予測が変わるかで、使われているかを見る（permutation importance［1］と同じ考え方）', 'Q2-b'],
    ['マーカーの効果', '同じ画像にマーカーを付けた版と外した版の logit 差', 'AUROC は病変そのものの特徴と混ざる。植え込んだ規則を学んだかを、反実仮想で直接測る', 'Q3'],
    ['ARI・NMI', '2 つの区分の一致度（0 は無関係、1 は完全一致）', 'クラスタの評価で標準の指標。作った区分が、植え込んだ少数側やラベルをどれだけ写しているかを測る', 'Q2-b × c・Q3'],
  ], [1.6, 2.3, 4.0, 1.1], { size: 9 });
  refs(s, '［1］Breiman, Machine Learning 2001（Random Forests の変数重要度）');
}

// 7. Q1 設定
{
  const s = slide('Q1 実験設定：同じ予測を切り方 × 指標で監査する', 'Q1 発見', '偏りが属性の軸に沿っているかを確かめるため、同じモデルの test 予測を属性セルと潜在スライスで切り、3 種類の偏りを測った');
  intent(s, '属性セルの監査で「偏りなし」と出ても、属性に沿わない偏りが隠れていないかを確かめる。隠れているなら、潜在部分群を使う意味がある');
  kv(s, [
    ['監査するモデル', 'CheXpert の ResNet-50（train の半分で学習）。植え込みなしの版と、ペースメーカーに触れる陽性を学習データで 10% に間引いた版'],
    ['属性での切り方', '年齢 × 人種（8 群）、性別 × 年齢 × 人種（21 群）'],
    ['潜在での切り方', '埋め込みの GMM（ラベル不使用）、残差を当てる決定木、監査器（val の残差を当てる小さな MLP の予測で 32 分位）。対照はランダム 32 群と、既知の文脈（デバイス・撮影方向など）'],
    ['測るもの', '較正の偏り：群ごとの log O/E のばらつき（次ページ）。順位の偏り：最悪群 AUROC。しきい値の偏り：群間の TPR 差（最大 − 最小、equal opportunity の差）'],
    ['比べ方', '最悪値や差は群が小さいほど悪く出る。そこで群の大きさを保ったまま群を患者単位で入れ替えた帰無分布（200 回）と比べる。切り方は test 以外で決める'],
    ['追試', 'ISIC 2019 の施設間（BCN20000 で学習 → HAM10000 で監査）'],
  ]);
  node(s, 'Trained model (ResNet-50)\ntest predictions', 6.35, 2.2, 2.8, 0.5);
  node(s, 'Attribute cells\nsex × age × race', 6.0, 3.0, 1.65, 0.55, THEME_TINT);
  node(s, 'Latent slices\nGMM / residual auditor', 7.85, 3.0, 1.65, 0.55, THEME_TINT);
  arrow(s, 7.75, 2.7, 6.83, 3.0); arrow(s, 7.75, 2.7, 8.67, 3.0);
  node(s, 'Per group: log O/E, AUROC, TPR', 6.0, 3.85, 3.5, 0.45);
  arrow(s, 6.83, 3.55, 6.83, 3.85); arrow(s, 8.67, 3.55, 8.67, 3.85);
  node(s, 'Compare with shuffled groups (null)', 6.0, 4.45, 3.5, 0.35);
  arrow(s, 7.75, 4.3, 7.75, 4.45);
}

// 8. Q1 物差し
{
  const s = slide('Q1 物差し：log O/E を使う理由', 'Q1 発見', 'AUROC は群の中の順位しか見ないため、群ごとの予測の水準のずれを捉えられない。そこで標準の指標に log O/E を加えた');
  bullets(s, [
    ['log O/E = log（群で観測した陽性数 ÷ 群の予測確率の和）。0 なら予測どおり、正なら過少予測、負なら過大予測'],
    ['右図の 2 群は群内の順位が同じなので AUROC は等しい。しかし群 Y は陽性を半分しか見込んでいない'],
    ['後から足す補正（Q2-a）が直せるのは、この水準のずれ。直す対象を直接測る'],
    ['群ごとの較正のずれをなくすという考え方は、multicalibration［1］や Multiaccuracy［2］と同じ'],
    ['揺れへの対処：予測確率を val で較正し、正しく較正されていても出るばらつき（ラベルを予測確率から引き直した帰無分布）を差し引く', 0],
  ], { size: 11 });
  figure(s, 'q1_oe_illustration');
  refs(s, '［1］Hébert-Johnson et al., ICML 2018　［2］Kim et al. (Multiaccuracy-Boost), AIES 2019。右図の数値は説明用の例');
}

// 9. Q1 結果（CheXpert）
{
  const s = slide('Q1 結果：見える偏りの種類が切り方で分かれる', 'Q1 発見', '属性セルには順位としきい値の偏りが出るが、較正のずれは出ない。潜在スライスの監査器は逆に、較正のずれだけを強く示す');
  figure(s, 'q1_metrics_chexpert', 9.0 / 2.9, { x: 1.2, y: 1.45, w: 7.6 });
  bullets(s, [
    ['年齢 × 人種：最悪群 AUROC 0.745（ランダムな群なら 0.816）、TPR 差 0.42（同 0.13）。log O/E のばらつきは 0。監査器のスライスは log O/E 0.24 で、AUROC・TPR はランダム並み'],
    ['植え込んだペースメーカーの過少予測（log O/E +0.62）は、属性セルでは −0.08 と消え、GMM クラスタでは +0.57 と見える'],
    ['注意：スコアの分位で切ると群内 AUROC は 0.42 まで下がる。群内のスコアの幅が狭いほど AUROC は機械的に下がるので、潜在スライスの AUROC の低さにはこの分が混じりうる'],
  ], { y: 3.92, w: 9.0, h: 0.95, size: 9 });
}

// 10. Q1 結果（ISIC）
{
  const s = slide('Q1 結果：施設をまたぐ ISIC でも同じ分かれ方', 'Q1 発見', '性別・年齢では順位としきい値の偏りが出て、較正のずれは出ない。較正のずれは潜在スライスと部位で出る');
  figure(s, 'q1_metrics_isic', 9.0 / 2.9, { x: 1.2, y: 1.45, w: 7.6 });
  bullets(s, [
    ['性別：最悪群 AUROC 0.759（ランダムな群なら 0.779）、TPR 差 0.14（同 0.04）。GMM 16 と監査器は log O/E 0.38・0.75 で、AUROC・TPR はランダム並み'],
    ['読み：潜在スライスで見つかるのは、属性セルが見落とす「群ごとの予測の水準のずれ」。属性の最悪群 AUROC の差は属性セルでも見えていて、Q2 はこの差を潜在の振り分けで縮められるかを問う'],
    ['監査器の値は seed で 0.38〜1.02 と揺れる（3 seed 平均を表示）'],
  ], { y: 3.92, w: 9.0, h: 0.95, size: 9 });
  refs(s, '誤りスライスの発見［1］と同じく、較正の偏りの発見には潜在部分群が使える。［1］Eyuboglu et al. (Domino), ICLR 2022');
}

// 10. Q2-a 設定
{
  const s = slide('Q2-a 実験設定：監査セットで後から補正する', 'Q2 是正 ＞ a 事後補正', '見つけた偏りを後から直せるかを確かめるため、1 段目を凍結し、1 段目が見ていない監査セットで群ごとの補正を学んだ');
  intent(s, '展開先で小さなラベル付きデータ（監査セット）を集め、後から直す場面を想定する。潜在で振り分ける補正が、全体の補正や属性で振り分ける補正より偏りを直せるかを見る');
  kv(s, [
    ['1 段目', 'CheXpert の ResNet-50 を train の半分 A で学習し、凍結する'],
    ['提案の補正', 'logit に ΔW_k·h(x) + Δb_k を足す（h は 1 段目の特徴、ΔW は低ランク）。群 k は [h(x); e(x)] から振り分ける。e は胸部 X 線の基盤モデル RAD-DINO の埋め込み'],
    ['比べる補正', '全体で 1 つ、属性で振り分け、GMM クラスタ別、LHCF 型、Multiaccuracy 型、最終層の再学習、DFR 型（付録 B1〜B4・B12）'],
    ['植え込む偏り', 'マーカー：A では陽性の 50%・陰性の 5% に白い四角を描き、陽性の手がかりとして覚えさせる。ペースメーカー：その陽性を A で 10% に間引き、学び損ねさせる'],
    ['監査セット B', '大きさ 1,000〜7.8 万枚 × 偏りの残り方 0・0.5・1（1 は A と同じ偏り）。測るのは植え込んだ群の AUROC と log O/E'],
  ], { size: 8.5 });
  node(s, 'Stage 1: ResNet-50 trained on half A (frozen)', 6.0, 2.2, 3.5, 0.4, THEME_TINT);
  node(s, 'h(x): stage-1 features', 6.0, 2.85, 1.65, 0.42);
  node(s, 'e(x): RAD-DINO embedding', 7.85, 2.85, 1.65, 0.42);
  arrow(s, 6.83, 2.6, 6.83, 2.85);
  node(s, 'Router → group k', 6.0, 3.5, 3.5, 0.35);
  arrow(s, 6.83, 3.27, 6.83, 3.5); arrow(s, 8.67, 3.27, 8.67, 3.5);
  node(s, 'logit + ΔW_k·h(x) + Δb_k', 6.0, 4.05, 3.5, 0.35, THEME_TINT);
  arrow(s, 7.75, 3.85, 7.75, 4.05);
  node(s, 'trained on audit set B (unseen by stage 1)', 6.0, 4.5, 3.5, 0.3, 'FFFFFF', 'FFFFFF', 8);
}

// 11. Q2-a 結果
{
  const s = slide('Q2-a 結果：事後補正は条件が厳しい', 'Q2 是正 ＞ a 事後補正', '植え込んだ偏りは直るが、偏りのない大きな監査セットが要り、実データでの改善は小さい');
  bullets(s, [
    ['右図：ペースメーカーを間引いた 1 段目を、[h; e] で振り分ける補正で直したとき'],
    ['ショートカット（マーカー）は 1,000 枚で消えるが、学び損ねた部分集団（ペースメーカー）は 2 万枚以上要る'],
    ['監査セットに学習時と同じ偏りが残ると、柔軟な補正ほど偏りを強める（Multiaccuracy 型で log O/E 0.62 → 0.85）'],
    ['[h; e] の振り分けが他を上回るのは、偏りのない全量で直すときだけで、差も小さい（0.813 対 Multiaccuracy 型 0.809）'],
    ['実データの CheXpert では全体 AUROC +0.003、最悪群は区間が 0 をまたぐ。ISIC の施設間では最終層の再学習に負ける'],
  ]);
  figure(s, 'q2a_audit_set');
}

// 12. Q2-b 設定
{
  const s = slide('Q2-b 実験設定：hypernet に渡す条件だけを替える', 'Q2 是正 ＞ b E2E の条件付け', '部分群ごとに推論を変える仕組みが効くかを確かめるため、モデルの構成は固定し、hypernet に渡す条件だけを替えた');
  intent(s, '属性で条件付けた hypernet は、HAM10000 で ResNet を上回る。属性の代わりに潜在部分群を渡しても、同じだけの利得が出るかを見る');
  kv(s, [
    ['モデル', 'Spatial LoRA。条件 c から hypernet が低ランク（rank 4）の重みの差分を作り、ResNet-50 の stage3・stage4 の conv と分類層に足す（右図）。全体を E2E で学習'],
    ['潜在の条件', 'ランダム 8 区分、DINOv2 GMM 8（DINOv2 の埋め込みを GMM で 8 群）、ラベル直交 GMM 8（INLP でラベルの方向を除いてから GMM。付録 B18）、DINOv2 の埋め込みそのもの（PCA 16 次元）'],
    ['画像にない情報', '撮影施設（潜在が施設を回収できたときの上限）、属性（性別・年齢・部位）'],
    ['対照', '条件を定数にした同じ構成。パラメータが増えた効果を除く'],
    ['データ・測るもの', 'HAM10000（主）と FairVLMed、各 3 seed。全体・既知施設・external の AUROC、性別 × 年齢の最悪群 AUROC'],
  ], { size: 8.5 });
  const bx = 6.0, bw = 1.6, hx = 8.0, hw = 1.5;
  ['Image', 'Stages 1–2 (shared)', 'Stage 3 + ΔW', 'Stage 4 + ΔW', 'Classifier + ΔW'].forEach((t, i) => {
    node(s, t, bx, 2.2 + i * 0.52, bw, 0.36, i >= 2 ? THEME_TINT : 'FFFFFF');
    if (i) arrow(s, bx + bw / 2, 2.2 + i * 0.52 - 0.16, bx + bw / 2, 2.2 + i * 0.52);
  });
  node(s, 'Condition c\n(attributes / cluster id /\nembedding / constant)', hx, 2.2, hw, 0.62);
  node(s, 'Hypernet', hx, 3.1, hw, 0.36, ACCENT_TINT, ACCENT);
  arrow(s, hx + hw / 2, 2.82, hx + hw / 2, 3.1);
  [3.24, 3.76, 4.28].forEach((y) => arrow(s, hx, 3.28, bx + bw, y + 0.18));
  s.addText('low-rank ΔW (rank 4)', { x: hx - 0.1, y: 3.95, w: 1.7, h: 0.3, fontFace: F, fontSize: 7.5, color: MUTED, margin: 0 });
}

// 13. Q2-b 結果
{
  const s = slide('Q2-b 結果：潜在の条件は定数と区別できない', 'Q2 是正 ＞ b E2E の条件付け', '潜在部分群で条件付けても、条件を定数にした同じモデルと区別できない');
  bullets(s, [
    ['右図は全体 AUROC の、定数との差。青は潜在部分群、灰は画像にない情報'],
    ['潜在の 4 条件は、全体・既知施設・external・性別 × 年齢の最悪群のどれでも、定数との差が ±0.015 以内'],
    ['施設・属性も全体 AUROC では区間が 0 をまたぐ。有意なのは属性の既知施設 AUROC（+0.018）だけ'],
    ['FairVLMed（緑内障、眼底 SLO）でも、どの条件も定数と区別できない'],
  ]);
  figure(s, 'q2b_conditions');
}

// 14. Q2-b 原因
{
  const s = slide('Q2-b なぜ効かないか：条件が使われていない', 'Q2 是正 ＞ b E2E の条件付け', '画像から計算できる条件はネットワークにとって新しい情報にならず、学習後のモデルは条件をほとんど使わない');
  bullets(s, [
    ['診断：test の画像はそのままで、条件だけを画像の間で入れ替えて推論し、AUROC の低下を見る（使われていない条件なら下がらない）'],
    ['潜在の条件の低下は 0.001〜0.005 で、意味のないランダム区分と同程度。属性は 0.069、施設は 0.025 下がる'],
    ['潜在部分群は画像の関数なので、backbone の中で同じものを計算できる。hypernet の経路を使う理由がない'],
    ['GroupDRO と組むと条件は使われるようになるが、変わるのは予測の水準だけで、AUROC は動かない'],
    ['属性で条件付ける既存研究の評価には、この入れ替えの対照がない'],
  ]);
  figure(s, 'q2b_shuffle');
}

// 15. Q2-b 属性の利得
{
  const s = slide('Q2-b 属性の利得はどこから来るか', 'Q2 是正 ＞ b E2E の条件付け', '属性の利得は分類層に属性を渡すだけで再現し、特徴抽出を属性で変える必要はない');
  bullets(s, [
    ['対照：同じ属性で、分類層だけを変調するモデル（backbone は素の ResNet-50）を 3 seed 回した'],
    ['分類層だけを属性で変調すると全体 0.918・external 0.871。backbone まで変調すると 0.910・0.822'],
    ['既知施設と性別 × 年齢の最悪群は同じ水準で、seed のばらつきは 1/5 に縮む'],
    ['external では属性が欠け、学習分布の外の値が backbone の特徴そのものを崩す'],
    ['利得は、陽性の事前確率を属性ごとに変える「情報」の効果で説明できる'],
  ]);
  figure(s, 'q2b_where');
}

// 16. Q2-c 設定
{
  const s = slide('Q2-c 実験設定：群を使う公平性損失 11 種', 'Q2 是正 ＞ c 公平性損失', '群を損失に入れる手法が属性の公平性を上げるかを確かめるため、仕組みの違う 11 手法を同じ条件で比べた');
  intent(s, 'LHCF は「潜在のコホート＋最悪コホート損失」で公平性が上がると報告した。GroupDRO 1 つに頼らず、群の扱い方が違う手法を広く試す');
  s.addTable([
    ['仕組み', '手法', '何をするか', '説明'].map((t) => ({ text: t, options: { bold: true, color: 'FFFFFF', fill: { color: THEME } } })),
    ...[
      ['群の重みを動かす', 'GroupDRO', '損失の高い群を重くする', 'B5'],
      ['', 'FairDRO', 'クラスごとに、誤りの多い群を重くする', 'B8'],
      ['', 'minimax AUC', '最も低い群間の AUC を上げる', 'B9'],
      ['', '最悪コホート損失（LHCF）', 'CE に、平均損失が最も高い群の損失を足す', 'B12'],
      ['', 'FIS', '損失の分布が全体とずれた群を重くする', 'B13'],
      ['群の量を揃える', '群の再重み付け', '群ごとの平均損失を等しく扱う', 'B6'],
      ['', '群 × クラス均等抽出', '群とクラスの組を同数ずつ抽出する', 'B7'],
      ['群の差を制約する', 'EO 制約', '群の間で TPR・FPR を揃える', 'B10'],
      ['', 'FaMI', 'logit と群の相互情報量を罰する', 'B14'],
      ['', '属性の敵対的除去（DANN）', '特徴から属性を当てられなくする', 'B11'],
      ['群を使わない', 'SWAD', '学習中の重みを平均する', 'B15'],
    ].map((r) => r.map((t, j) => ({ text: t, options: j === 0 ? { bold: true, color: THEME } : {} }))),
  ], { x: 0.5, y: 2.12, w: 9, colW: [1.6, 2.3, 4.4, 0.7], fontFace: F, fontSize: 8.5, color: INK, border: { type: 'solid', pt: 0.5, color: RULE }, valign: 'middle', margin: [1, 4, 1, 4], rowH: 0.21 });
  outline.push('（表：11 手法）\n');
  refs(s, '共通：ResNet-50、HAM10000、30 epoch、3 seed。群は性別 × 年齢（4 群）、DINOv2 GMM 8、LHCF の隠れコホート（ERM の埋め込みの GMM、K = 11）。ハイパーパラメータは元論文の値（記載がなければ λ = 1）。checkpoint は val の全体 AUROC で選ぶ');
}

// 17. Q2-c 結果
{
  const s = slide('Q2-c 結果：どの公平性損失も最悪群を上げない', 'Q2 是正 ＞ c 公平性損失', '群を使う公平性損失を 11 種試しても、性別 × 年齢の最悪群 AUROC は ERM から有意に上がらない');
  bullets(s, [
    ['右図は性別 × 年齢の群で回した結果。GMM 8・LHCF のコホートで回しても同じ傾向'],
    ['群の重みは弱い群に向かないか、ほとんど動かない。GroupDRO の GMM 8 は、3 seed とも最悪群でないクラスタに寄る'],
    ['ラベルと絡んだ群に独立性を課すと判別が壊れる（FaMI × LHCF のコホートで全体 0.771、ERM は 0.899）'],
    ['有意に上がったのは SWAD ＋ LHCF の最悪コホート損失（+0.045）だけ'],
    ['MEDFAIR［1］も、公平性手法が ERM を有意に超えにくいと報告している'],
  ]);
  figure(s, 'q2c_methods', 4.6 / 3.5);
  refs(s, '［1］Zong et al. (MEDFAIR), ICLR 2023');
}

// 18. Q2-c LHCF
{
  const s = slide('Q2-c LHCF の利得の正体', 'Q2 是正 ＞ c 公平性損失', 'LHCF 型の利得は、群をクラス（良性・悪性）に替えても同じだけ出る。隠れコホートの発見ではなく、難しい層を重くした効果である');
  bullets(s, [
    ['対照：SWAD ＋ 最悪コホート損失の「群」だけを、クラス・性別 × 年齢・大きさだけ揃えたランダムなコホートに替えた'],
    ['LHCF のコホート（K = 11）は 8 つが悪性率ほぼ 0、2 つが 0.87 以上で、悪性度で切れている'],
    ['損失が残るのは境界のコホートだけなので、最悪コホート損失は「迷う例を重くする」損失として働く'],
    ['クラスを群にしたときとの差は +0.010 [−0.010, +0.034]。ランダムなコホートだけが SWAD 単独の水準に落ちる'],
    ['発見の指標で偏りが大きく見える区分ほど、是正では悪くなる（順位相関 −0.85、付録 A1）'],
  ]);
  figure(s, 'q2c_lhcf');
}

// 19. Q2 b × c 設定
{
  const s = slide('Q2-b × c 実験設定：hypernet ＋ LHCF 型の損失', 'Q2 是正 ＞ b × c', 'Q2-c は ResNet だけで比べたので、hypernet と組めば結論が変わるかを 2 × 2 で確かめた');
  intent(s, '条件付け（振り分け）と群の損失を同時に使うと、どちらか単独では出ない利得が出るかを見る');
  kv(s, [
    ['モデル', 'Spatial LoRA（Q2-b と同じ構成、E2E）'],
    ['条件', '定数、または LHCF の隠れコホート（11 群）'],
    ['損失', 'ERM、または SWAD ＋ 最悪コホート損失（群は LHCF のコホート）'],
    ['データ・測るもの', 'HAM10000、各 3 seed。全体・external・性別 × 年齢の最悪群 AUROC。コホートとラベルの NMI（コホートがラベルをどれだけ写すか）'],
  ], { size: 9 });
  const gx = 5.95, gy = 2.3, cw = 1.4, lw = 0.75, rh = 0.7;
  node(s, 'ERM', gx + lw, gy, cw, 0.4, THEME_TINT);
  node(s, 'SWAD + worst-\ncohort loss', gx + lw + cw, gy, cw, 0.4, THEME_TINT);
  node(s, 'Constant', gx, gy + 0.4, lw, rh, THEME_TINT);
  node(s, 'LHCF\ncohort', gx, gy + 0.4 + rh, lw, rh, THEME_TINT);
  for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) node(s, 'Spatial LoRA\n3 seeds', gx + lw + j * cw, gy + 0.4 + i * rh, cw, rh);
  s.addText('rows: conditioning variable / columns: loss', { x: gx, y: gy + 0.4 + 2 * rh + 0.1, w: 3.55, h: 0.3, fontFace: F, fontSize: 7.5, color: MUTED, margin: 0 });
}

// 20. Q2 b × c 結果
{
  const s = slide('Q2-b × c 結果：コホートを条件にするとラベルが漏れる', 'Q2 是正 ＞ b × c', '隠れコホートを hypernet の条件に渡すと、train でラベルが漏れ、全体 AUROC が 0.03〜0.05 下がる');
  bullets(s, [
    ['LHCF のコホートは、ERM が自分の train の埋め込みから作る。ERM は train をほぼ分離しているので、train ではコホートがラベルをよく表す'],
    ['モデルは train でコホートを「ラベルのヒント」として使い、test でそのヒントが外れる（全体 −0.032、SWAD ＋ 最悪コホート損失では −0.049）'],
    ['区分を作るモデルが見ていない画像で作ると、NMI は train と test で揃い、悪化は −0.006 まで縮む'],
    ['SWAD ＋ 最悪コホート損失の利得は、Spatial LoRA でも ResNet と同じ大きさ（全体 +0.022）で、振り分けても増えない'],
  ]);
  figure(s, 'q2_leak');
}

// 21. Q3-a 設定
{
  const s = slide('Q3-a 実験設定：規則が逆になる部分群を植え込む', 'Q3 条件 ＞ a 正の対照', '否定的な結果の原因が仕組みの欠陥か、データに規則の違いがないことかを分けるため、部分群で規則が逆になるデータを作った');
  intent(s, 'Q2-b で潜在の条件が使われなかったのは、振り分けが働かないからか、HAM10000 に部分群ごとの規則の違いがないからか。違いを人工的に作って確かめる');
  kv(s, [
    ['植え込み', 'HAM10000 の画像の隅に、右図の規則で白い四角（マーカー）を描く。train・val・test に同じ規則を当てる'],
    ['少数側 B', 'DINOv2 GMM 8 のクラスタ 2・4。GMM 8 で条件付けたモデルだけが、B で規則を切り替える手がかりを持つ'],
    ['比べる条件', 'Spatial LoRA の条件を、定数・ランダム 8 区分・GMM 8 に替える（各 3 seed）。付与率 50% の弱い版と 90% の強い版'],
    ['測るもの', 'B の AUROC と、マーカーの効果（同じ画像にマーカーを付けた版と外した版の logit 差。A で正・B で負なら規則どおり）'],
    ['目安', '植え込み前の ERM の得点にマーカーの情報をロジスティック回帰で足し、規則を知る場合と知らない場合の到達点を作る'],
  ], { size: 8.5 });
  box(s, 'Majority A: GMM 8 clusters 0, 1, 3, 5, 6, 7\nmarker on 90% of malignant, 10% of benign\n→ marker means malignant',
    { x: 6.0, y: 2.2, w: 3.5, h: 0.9, fill: THEME_TINT, line: THEME, size: 8.5, color: INK });
  box(s, 'Minority B: GMM 8 clusters 2, 4\n(12% of benign, 27% of malignant)\nmarker on 10% of malignant, 90% of benign\n→ marker means benign',
    { x: 6.0, y: 3.2, w: 3.5, h: 1.0, fill: ACCENT_TINT, line: ACCENT, size: 8.5, color: INK });
  s.addText('strong version (90%) shown', { x: 6.0, y: 4.25, w: 3.5, h: 0.25, fontFace: F, fontSize: 7.5, color: MUTED, margin: 0 });
  refs(s, 'マーカーは画像上の手がかりへのショートカット学習（Geirhos et al., Nat. Mach. Intell. 2020; DeGrave et al., Nat. Mach. Intell. 2021）を、部分群ごとに向きを変えて植えた形');
}

// 22. Q3-a 結果
{
  const s = slide('Q3-a 結果：境界に合った区分なら振り分けは効く', 'Q3 条件 ＞ a 正の対照', '境界を含む区分で条件付けたモデルだけが B の規則を反転させ、B の AUROC を定数より +0.094 上げた');
  bullets(s, [
    ['B でのマーカーの効果：GMM 8 は 3 seed とも負（−3.6、−2.4、−0.9）、定数とランダム区分は正'],
    ['ランダム 8 区分は定数と変わらない（−0.022、区間が 0 をまたぐ）。容量が増えた効果ではない'],
    ['多数側 A も +0.030 上がる。B の例外に引きずられず、A の規則を強く使える'],
    ['手がかりが弱い版（付与率 50%）では、反転は起きても AUROC はほぼ動かない'],
    ['実データで潜在の条件が使われなかったのは、仕組みの欠陥ではなく、場に規則の違いがないためと読める'],
  ]);
  figure(s, 'q3a_rule');
}

// 23. Q3-b 設定
{
  const s = slide('Q3-b 実験設定：1 段目を凍結した 2 段学習', 'Q3 条件 ＞ b 1 段か 2 段か', '振り分けを後付けの補正にしても効くかを確かめるため、train を二分し、凍結した 1 段目の上で条件付きの部分だけを学んだ');
  intent(s, 'E2E では特徴と振り分けを一緒に学ぶ。凍結した 1 段目の上の補正だけで少数側の規則を直せるかを見る。2 段にすると、1 段目が見ていない画像の誤りを Q3-c の区分づくりに使える');
  kv(s, [
    ['train の分け方', '病変単位で、良性・悪性それぞれを半分ずつ stage1（4,228 枚）と stage2（4,229 枚）に分ける。val・test は同じ'],
    ['1 段目', 'ResNet-50 ERM を stage1 だけで学び、凍結する（BatchNorm も固定）'],
    ['2 段目', 'Spatial LoRA と hypernet だけを stage2 で学ぶ。条件は定数・ランダム 8 区分・GMM 8'],
    ['データ・測るもの', 'Q3-a の強い版で、B の AUROC とマーカーの効果。実データでも、LHCF のコホートで条件付けた版と最悪コホート損失の版を回した'],
  ], { size: 9 });
  node(s, 'HAM10000 train (8,457 images)', 6.0, 2.2, 3.5, 0.38, THEME_TINT);
  node(s, 'stage1 half (4,228)', 6.0, 2.85, 1.65, 0.38);
  node(s, 'stage2 half (4,229)', 7.85, 2.85, 1.65, 0.38);
  arrow(s, 7.0, 2.58, 6.83, 2.85); arrow(s, 8.5, 2.58, 8.67, 2.85);
  node(s, 'Stage 1: ResNet-50 ERM\n→ frozen', 6.0, 3.5, 1.65, 0.55, THEME_TINT);
  node(s, 'Stage 2: Spatial LoRA\n+ hypernet', 7.85, 3.5, 1.65, 0.55, ACCENT_TINT, ACCENT);
  arrow(s, 6.83, 3.23, 6.83, 3.5); arrow(s, 8.67, 3.23, 8.67, 3.5); arrow(s, 7.65, 3.78, 7.85, 3.78);
  node(s, 'condition: constant /\nrandom 8 / GMM 8', 7.85, 4.25, 1.65, 0.5);
  arrow(s, 8.67, 4.25, 8.67, 4.05);
}

// 24. Q3-b 結果
{
  const s = slide('Q3-b 結果：2 段でも効くが E2E に届かない', 'Q3 条件 ＞ b 1 段か 2 段か', '凍結した 1 段目に条件付きの補正を足す 2 段学習でも効くが、到達点は E2E に届かない');
  bullets(s, [
    ['GMM 8 は定数より B で +0.123 [+0.080, +0.163]。ただし B での反転の大きさは E2E の数分の 1'],
    ['条件を持たない 2 段目は、B を 1 段目より悪くする（0.771 → 0.738）。多数側の規則がさらに強まる'],
    ['CheXpert でも、同時学習では補正が backbone 側へ移った。発見が目的なら凍結、性能が目的なら E2E という使い分けになる'],
    ['実データでは、LHCF のコホートで条件付けても改善しない（全体 −0.006）'],
  ]);
  figure(s, 'q3b_stages');
}

// 25. Q3-c 設定
{
  const s = slide('Q3-c 実験設定：誤りから区分を作り直す', 'Q3 条件 ＞ c 区分の作り直し', '実データでは規則の境界が分からないため、モデルの誤りから区分を作って条件付け、学び直す手順を 3 周回した');
  intent(s, 'Q3-a では区分を境界に合わせて作っていた。境界を知らなくても、誤りの集まりから境界に近い区分を作れるかを見る。誤りから群を推す JTT・EIIL（付録 B16・B17）と同じ発想');
  kv(s, [
    ['① 残差', '1 段目が見ていない stage2 の画像で、残差 r = p − y を求める（p は予測確率、y は正解）'],
    ['② 区分', '1 段目の埋め込み z（PCA 32 次元）に残差を掛けた g = r·z を、GMM で 8 群に分ける。誤りの大きさと向きが近い画像が同じ群に入る'],
    ['③ 割り当て', '区分は正解 y を使うので、そのままでは test に使えない。z から区分を当てるロジスティック回帰（振り分け係）で全 split に割り当てる'],
    ['④ 学び直し', 'その区分で条件付けた 2 段目を学び、その残差で①に戻る。3 周'],
    ['測るもの', 'B の AUROC とマーカーの効果、区分と B の一致度（ARI）。実データでも同じ手順を回した'],
  ], { size: 8.5 });
  node(s, 'Model\n(stage 1 at iteration 1)', 6.0, 2.25, 1.65, 0.6, THEME_TINT);
  node(s, 'Residual r = p − y\non held-out images', 7.85, 2.25, 1.65, 0.6);
  node(s, 'GMM 8 on g = r · z', 7.85, 3.45, 1.65, 0.6);
  node(s, 'Router (z → partition)\nassigns all splits', 6.0, 3.45, 1.65, 0.6);
  arrow(s, 7.65, 2.55, 7.85, 2.55); arrow(s, 8.67, 2.85, 8.67, 3.45); arrow(s, 7.85, 3.75, 7.65, 3.75); arrow(s, 6.83, 3.45, 6.83, 2.85);
  s.addText('retrain stage 2 conditioned on the partition (× 3 iterations)', { x: 6.0, y: 4.2, w: 3.5, h: 0.3, fontFace: F, fontSize: 7.5, color: MUTED, align: 'center', margin: 0 });
}

// 26. Q3-c 結果
{
  const s = slide('Q3-c 結果：誤りからは境界を回収できない', 'Q3 条件 ＞ c 区分の作り直し', 'モデルの誤りから区分を 3 周作り直しても、植え込んだ境界は回収できなかった');
  bullets(s, [
    ['区分と B の一致（ARI）は 0.035 → 0.010 → 0.008。B が過半を占める区分は 1 つもできない'],
    ['原因：B の陰性では、マーカーの有無で残差がほぼ同じ（+0.292 と +0.286）。規則の反転が「どの画像を間違えるか」に表れない'],
    ['そのため区分は残差の符号で切れ、埋め込みからは再現できない（振り分け係の一致 0.49〜0.56）'],
    ['実データでも、性別 × 年齢・全体・external のどれも改善しない'],
  ]);
  figure(s, 'q3c_iter');
}

// 27. まとめ
{
  const s = slide('まとめ', 'まとめ', '潜在部分群は較正の偏りの発見には使えるが、是正には効かない。振り分けが効く条件は分かったが、実データでその条件を満たす方法がまだない');
  pyramid(s, [
    ['一部支持', '較正のずれは潜在だけに出る。AUROC・TPR 差は属性にも出る'],
    ['条件つき', '植え込みは直る。実データは +0.003'],
    ['不支持', '条件は使われない。利得は画像外の情報'],
    ['不支持', '11 手法とも有意差なし。LHCF はクラスの重み付け'],
    ['支持', '少数側 B が +0.094'],
    ['支持（弱い）', 'B が +0.123。E2E に届かない'],
    ['不支持', 'B との ARI は 0.035 以下'],
  ]);
}

// 28. 論文の筋と次の一手
{
  const s = slide('論文の筋と次の一手', '今後', '「属性を使わない部分群の条件付けは情報を足さない」を、効く条件と評価の対照つきで書く筋を提案する');
  bullets(s, [
    ['主張 1：潜在部分群は属性セルに見えない較正の偏りの発見には有効だが、補正・条件付け・公平性損失のどれでも属性の公平性を改善しない（4 データセット）'],
    ['主張 2：条件付きモデルの利得は、条件が運ぶ画像外の情報から来る。振り分けが効くのは、部分群で規則が食い違う場に限られる（正の対照）'],
    ['貢献にできる評価の対照：条件の入れ替え、分類層だけの条件付け、クラスを群にした同じ損失、大きさを揃えたランダムなコホート、held-out での区分づくり'],
    ['最も弱い所：規則の境界を実データで見つける方法がない'],
    ['次の一手（提案）：規則の食い違いが実際にありそうな場（施設ごとに診断の確定方法や撮影機器が違うデータなど）で、異質性を先に測る'],
    ['議論したいこと：否定的な結果を軸にしたこの筋で書くか、正の対照を手法の側に広げるか'],
  ], { w: 9, size: 12.5 });
}

// 付録 A：補足の結果
{
  const s = slide('付録 A1 発見の指標と是正の効果', '付録 A', '偏りを大きく見せる区分は難しさとラベルで切れていて、是正に使うと性能を下げる');
  bullets(s, [
    ['発見：val での全体 AUROC と群内 AUROC の差。是正：群 × クラス等重みでの最終層の再学習（DFR 型）と、群を使わない再学習の差'],
    ['得点の分位や ERM 特徴の k-means は偏りを大きく見せるが、易しい例と難しい例の比率を変えるだけになる'],
    ['属性・施設・DINOv2 のクラスタは、是正しても ±0.005 以内'],
    ['緩和に使う部分群の選び方で結果が大きく変わる、という報告［1］と整合する'],
  ]);
  figure(s, 'app_discovery', 4.8 / 3.3);
  refs(s, '［1］Alloula et al. (Subgroups Matter), ICML 2025');
}
{
  const s = slide('付録 A2 公平性手法の比較（HAM10000）', '付録 A', 'ERM との差（seed 平均、病変単位の bootstrap 95% 区間）。群は性別 × 年齢');
  table(s, ['手法', 'Δ 全体', 'Δ external', 'Δ 性別 × 年齢の最悪群'], [
    ['GroupDRO', '+0.002 [−0.010, +0.015]', '+0.009 [−0.029, +0.042]', '+0.012 [−0.027, +0.044]'],
    ['群の再重み付け', '−0.006 [−0.018, +0.006]', '−0.016 [−0.046, +0.012]', '+0.014 [−0.039, +0.043]'],
    ['群 × クラス均等抽出', '+0.012 [+0.001, +0.024]', '+0.015 [−0.023, +0.051]', '+0.018 [−0.019, +0.048]'],
    ['FairDRO', '+0.002 [−0.013, +0.017]', '+0.007 [−0.034, +0.041]', '+0.017 [−0.038, +0.043]'],
    ['minimax AUC', '+0.014 [+0.002, +0.027]', '+0.011 [−0.017, +0.039]', '+0.022 [−0.014, +0.049]'],
    ['EO 制約', '−0.030 [−0.049, −0.011]', '−0.066 [−0.115, −0.019]', '−0.030 [−0.136, +0.003]'],
    ['属性の敵対的除去（DANN）', '+0.006 [−0.005, +0.018]', '−0.012 [−0.040, +0.014]', '+0.026 [−0.013, +0.049]'],
    ['最悪コホート損失', '+0.008 [−0.003, +0.020]', '+0.005 [−0.030, +0.038]', '+0.040 [−0.018, +0.073]'],
    ['FIS', '−0.006 [−0.018, +0.005]', '−0.009 [−0.038, +0.020]', '−0.002 [−0.067, +0.034]'],
    ['FaMI', '−0.003 [−0.019, +0.012]', '−0.012 [−0.047, +0.022]', '+0.023 [−0.051, +0.050]'],
    ['SWAD（群なし）', '+0.014 [+0.007, +0.022]', '+0.001 [−0.024, +0.026]', '+0.021 [−0.023, +0.040]'],
    ['SWAD ＋ 最悪コホート損失（LHCF のコホート）', '+0.022 [+0.012, +0.034]', '+0.013 [−0.026, +0.054]', '+0.045 [+0.008, +0.081]'],
  ], [2.7, 2.1, 2.1, 2.1], { hi: [11], size: 8.5 });
  refs(s, 'ERM：全体 0.899、external 0.850、性別 × 年齢の最悪群 0.849（val の全体 AUROC で選択）。3 seed（42〜44）');
}
{
  const s = slide('付録 A3 Spatial LoRA × LHCF 型の損失（E2E）', '付録 A', 'Spatial LoRA でも効くのは SWAD ＋ 最悪コホート損失だけで、コホートを条件にすると悪くなる');
  table(s, ['モデル', '全体', 'external', '性別 × 年齢の最悪群', 'GMM 8 の最悪群', 'LHCF のコホートの最悪'], [
    ['ResNet-50 ERM', '0.899', '0.850', '0.849', '0.772', '0.709'],
    ['ResNet-50 SWAD ＋ 最悪コホート損失', '0.921', '0.863', '0.894', '0.835', '0.800'],
    ['Spatial LoRA：定数、ERM', '0.900', '0.840', '0.863', '0.803', '0.737'],
    ['Spatial LoRA：定数、SWAD ＋ 最悪コホート損失', '0.922', '0.879', '0.887', '0.817', '0.767'],
    ['Spatial LoRA：コホートで条件付け、ERM', '0.868', '0.811', '0.833', '0.764', '0.673'],
    ['Spatial LoRA：コホートで条件付け、SWAD ＋ 最悪コホート損失', '0.873', '0.839', '0.846', '0.783', '0.702'],
  ], [3.4, 0.9, 0.9, 1.3, 1.2, 1.3], { size: 9.5 });
  refs(s, 'val の全体 AUROC で選択（SWAD は重みが 1 つ）。3 seed 平均');
}
{
  const s = slide('付録 A4 誤りから作り直した区分の性質', '付録 A', 'どの周の区分も B を捉えず、条件として効かない');
  table(s, ['周', 'Δ AUROC B（規則の場、2 段目の定数との差）', 'マーカーの効果 B', 'B との ARI（test）', 'B が過半の区分の B 被覆率', 'Δ 性別 × 年齢の最悪群（実データ）'], [
    ['1 周目', '−0.001 [−0.022, +0.017]', '+1.01', '0.035', '0', '+0.002 [−0.014, +0.016]'],
    ['2 周目', '+0.012 [−0.005, +0.030]', '+0.99', '0.010', '0', '−0.003 [−0.014, +0.023]'],
    ['3 周目', '+0.001 [−0.014, +0.018]', '+0.94', '0.008', '0', '−0.002 [−0.015, +0.012]'],
    ['（参考）GMM 8（境界を含む）', '+0.123 [+0.080, +0.163]', '−0.61', '—', '—', '—'],
  ], [1.6, 2.0, 1.1, 1.1, 1.4, 1.8], { size: 9.5 });
  refs(s, '区分は実データで撮影施設と NMI 0.18〜0.25、性別 × 年齢とは 0.03');
}

// 付録 B：先行手法の説明（1 手法 1 枚）
function method(id, name, message, rows, source) {
  const s = slide(`付録 ${id} ${name}`, '付録 B 手法の説明', message);
  kv(s, rows, { y: 1.7, w: 9, labelW: 1.6, size: 12 });
  refs(s, `出典：${source}`);
}
method('B1', 'Multiaccuracy-Boost', '小さな監査セットで残差の大きい領域を探し、そこの予測を少しずつ直すことを繰り返す事後補正', [
  ['考え方', '属性を使わない監査器（埋め込みからの回帰）で、予測の残差が系統的に大きい領域を探す。見つかった領域の logit を少し動かし、また探す。黒箱のモデルのまま、監査器が使わない属性の群の誤りも減ると報告されている'],
  ['更新', 'f_{t+1}(x) = f_t(x) − η·a_t(x)。a_t は残差 f_t(x) − y を当てる監査器。残差が監査器で当てられなくなったら止める'],
  ['本研究での使い方', 'Q2-a の比較相手（Multiaccuracy 型）。監査器は [h; e] の主成分上の浅い回帰木。改善しないときは早期停止で元のモデルに留まる'],
  ['見たかったこと', '提案の振り分け補正と同じ「監査セットで後から直す」枠組みの代表。構図がほぼ同じなので、必須の比較相手'],
], 'Kim, Ghorbani & Zou, "Multiaccuracy: Black-Box Post-Processing for Fairness in Classification", AIES 2019');
method('B2', '最終層の再学習（LLR）', '特徴抽出器は固定し、最終層だけを別のデータで学び直す', [
  ['考え方', '特徴は十分に良く、偏りは最終層の重み付けにあるという見立て。held-out のデータで最終層（ロジスティック回帰）を学び直すだけで、群ラベルなしでも最悪群が改善すると報告されている'],
  ['本研究での使い方', 'Q2-a の比較相手。1 段目の特徴 h(x) の上で、監査セットを使ってロジスティック回帰を学び直す。群は使わない'],
  ['見たかったこと', '群を作る補正（提案）が、群を作らない最も単純な補正に勝てるか。ISIC の施設間では、これが最良だった'],
], 'LaBonte et al., "Towards Last-layer Retraining for Group Robustness with Fewer Annotations", NeurIPS 2023');
method('B3', 'DFR（Deep Feature Reweighting）', '群が均衡した held-out データで最終層だけを学び直し、ショートカットへの依存を減らす', [
  ['考え方', '特徴抽出器は、ショートカットがあっても本質的な特徴も学んでいる。群ラベル付きの群均衡なデータで最終層を学び直せば、本質的な特徴に重みが移る'],
  ['本研究での使い方', 'Q2-a の比較相手（DFR 型）。年齢 × 人種 × クラスの各セルを等重みにして最終層を学び直す。付録 A1 では、同じ再学習で群の切り方を 10 通り替えた'],
  ['見たかったこと', '属性を使う最終層の補正の代表として、潜在で振り分ける補正と比べる'],
], 'Kirichenko, Izmailov & Wilson, "Last Layer Re-Training is Sufficient for Robustness to Spurious Correlations", ICLR 2023');
method('B4', 'GEORGE（潜在クラスタで群を作る）', '特徴空間をクラスタリングして隠れた部分クラスを見つけ、それを群として学び直す', [
  ['考え方', 'ERM の特徴をクラスごとにクラスタリングすると、ラベルでは見えない部分クラス（隠れた層）が群として現れる。その群で GroupDRO を回すと、群ラベルなしで最悪群が上がる'],
  ['本研究での使い方', '「埋め込みのクラスタを群にする」発想を全体で使った（Q1 の GMM、Q2-a の GMM クラスタ別補正、Q2-c の GMM 8）。GEORGE と違い、クラスタリングにラベルを使わない'],
  ['見たかったこと', '潜在部分群の作り方の基準。LHCF もこの流れにある'],
], 'Sohoni et al., "No Subclass Left Behind: Fine-Grained Robustness in Coarse-Grained Classification Problems", NeurIPS 2020');
method('B5', 'GroupDRO', '最も損失の高い群を重くして学習し、最悪群の損失を下げる', [
  ['考え方', '群ごとの平均損失 L_g の重み付き和を最小化する。重み q は、損失の高い群ほど大きくなるよう学習中に更新する'],
  ['式', 'min_θ max_q Σ_g q_g·L_g(θ)。q は指数勾配で q_g ← q_g·exp(η·L_g) と更新し、正規化する'],
  ['本研究での使い方', 'Q2-c。q の学習率 η = 0.1、class weight 付き CE。群は性別 × 年齢・GMM 8'],
  ['見たかったこと', '最悪群を狙う手法の代表。学習の終わりに q が実際に弱い群へ寄ったかも確かめた（寄らなかった）'],
], 'Sagawa et al., "Distributionally Robust Neural Networks for Group Shifts", ICLR 2020');
method('B6', '群の再重み付け', '群ごとの平均損失を等しい重みで足し、少数の群も多数の群と同じだけ効かせる', [
  ['考え方', '標本の平均ではなく、群ごとの平均損失を等しく平均する。標本数の少ない群の損失が、多い群に埋もれなくなる'],
  ['式', 'L = (1/G) Σ_g L_g'],
  ['本研究での使い方', 'Q2-c。群は性別 × 年齢・GMM 8'],
  ['見たかったこと', '重みを学習しない最も単純な群の均衡化。こうした単純な均衡化が最悪群の改善で強い比較相手になる、と報告されている'],
], 'Idrissi et al., "Simple Data Balancing Achieves Competitive Worst-Group-Accuracy", CLeaR 2022');
method('B7', '群 × クラス均等抽出', 'ミニバッチを（群, クラス）の組が同数になるよう抽出する', [
  ['考え方', '群の大きさとクラスの比の両方を、抽出の段階で揃える。どの群でもバッチ内の陽性率が同じになる'],
  ['本研究での使い方', 'Q2-c。群は性別 × 年齢・GMM 8'],
  ['見たかったこと', '損失を変えずにデータの見せ方だけで均衡化する比較相手（B6 と同じ系統）'],
], 'Idrissi et al., "Simple Data Balancing Achieves Competitive Worst-Group-Accuracy", CLeaR 2022');
method('B8', 'FairDRO', 'クラスごとに、誤りの多い群を重くする重みを最悪化して学習する', [
  ['考え方', '正解クラス y ごとに群の重み q^y を持ち、群別損失の重み付き和を最小化する。q^y は、平均から外れた群ほど重くなる範囲（χ² 球）の中で最悪のものを取る。クラスを条件にすることで、equalized odds に近い公平性を狙う'],
  ['本研究での使い方', 'Q2-c。q は train 全件の誤り率から epoch ごとに更新し、バッチは群 × クラス均等抽出。群は性別 × 年齢・GMM 8'],
  ['見たかったこと', 'GroupDRO のクラスを分けた版。陽性率の違う群を同じ物差しで比べない設計'],
], 'Jung et al., "Re-weighting Based Group Fairness Regularization via Classwise Robust Optimization", ICLR 2023');
method('B9', 'minimax AUC', '群の組ごとの AUC 損失を最悪化して下げ、最も低い群間 AUC を上げる', [
  ['考え方', '陽性が群 g、陰性が群 h の組ごとに順位の損失（陽性が陰性より低く並ぶ度合い）を測り、組の重み q を最悪化して最小化する。群をまたいだ順位も含め、最悪の AUC を上げる'],
  ['本研究での使い方', 'Q2-c。q の初期値は train の陽性・陰性の群分布。対照として、q を初期値に固定した同じ AUC 損失も回した（q を学習した効果を切り分けるため）'],
  ['見たかったこと', '評価指標（AUROC）そのものを群ごとに最適化する手法。CE と違う損失の効果'],
], 'Yang et al., "Minimax AUC Fairness: Efficient Algorithm with Provable Convergence", AAAI 2023');
method('B10', 'EO 制約（proxy-Lagrangian）', '群の間で TPR と FPR が揃うことを制約にし、ラグランジュ乗数で解く', [
  ['考え方', 'equalized odds（群の間で TPR と FPR が等しい）を制約として課す。率は微分できないので、制約の判定には実際の率を、モデルの更新には滑らかな代理を使う'],
  ['本研究での使い方', 'Q2-c。全ての群の組で率の差に上限を置く。代理関数は smoothstep（研究用の設計で、原著の指定ではない）'],
  ['見たかったこと', '損失を重み付けるのではなく、群の間の差を直接制約する系統の代表'],
], 'Cotter et al., "Optimization with Non-Differentiable Constraints with Applications to Fairness, Recall, Churn, and Other Goals", JMLR 2019');
method('B11', '属性の敵対的除去（DANN）', '特徴から属性を当てる敵対者を置き、特徴を「属性を当てられない」方向に学ぶ', [
  ['考え方', '特徴の上に属性を当てる分類器（敵対者）を置き、勾配反転層を挟んで学習する。敵対者は属性を当てようとし、特徴抽出器は当てられないようにする'],
  ['本研究での使い方', 'Q2-c。敵対者は性別（カテゴリ）と年齢（連続）を当てる。属性の群でのみ回した'],
  ['見たかったこと', '群の重みでも制約でもなく、表現から属性を消す系統の代表'],
], 'Ganin et al., "Domain-Adversarial Training of Neural Networks", JMLR 2016');
method('B12', 'LHCF（最悪コホート損失）', '学習済みモデルの埋め込みの GMM で隠れコホートを作り、最も損失の高いコホートを重くする', [
  ['考え方', '属性を使わず、見た目の似た隠れコホートで公平性を改善する。コホートは学習済み分類器の埋め込みに GMM を当てて作り、数 K は BIC で選ぶ。SWAD と組み合わせる'],
  ['式', 'ℓ_wst = CE + λ·max_k R_k。R_k はバッチ内のコホート k の平均損失'],
  ['本研究での使い方', 'Q2-c・Q2-b × c。ERM（ResNet-50）の train 埋め込みを PCA 32 次元にして GMM、K = 11。λ は本文に値がないので 1。元論文は ResNet-18'],
  ['見たかったこと', '「潜在のコホート＋公平性損失」の直接の先行研究。利得がコホートの発見から来るかを、群をクラスやランダムに替えて確かめた'],
], 'Masroor et al., "Fairness Beyond Demographics" (LHCF), MICCAI 2026');
method('B13', 'FIS（Fair Identity Scaling）', '損失の高い標本と、損失の分布が全体とずれた群を重くする', [
  ['考え方', '標本の重み（バッチ内の損失の softmax）と、群の重み（群の損失分布と全体の損失分布の Sinkhorn 距離の softmax）を、比 c で混ぜて CE に掛ける'],
  ['本研究での使い方', 'Q2-c。c = 0.5（論文の既定）。論文は群ごとに別のバッチを取るが、本研究はバッチ内のその群の標本で代えた。群は性別 × 年齢・LHCF のコホート'],
  ['見たかったこと', 'LHCF が比較に並べた手法。医用画像の公平性で使われる重み付けの代表'],
], 'Luo et al., "FairVision: Equitable Deep Learning for Eye Disease Screening via Fair Identity Scaling", arXiv 2023');
method('B14', 'FaMI', 'logit と群の相互情報量を罰し、予測の分布を群の間で揃える', [
  ['考え方', '予測（logit）と群の相互情報量を損失に足す。相互情報量の勾配は、群ごとの密度と全体の密度のスコア関数の差で推定する'],
  ['本研究での使い方', 'Q2-c。重み μ = 0.5（論文の値）。論文の多項式カーネルの推定器は式がないので、1 次元の logit 差へのガウス KDE で代えた。群は性別 × 年齢・LHCF のコホート'],
  ['見たかったこと', 'LHCF が比較に並べた手法。群と予測の独立性を課す系統。コホートがラベルと絡むと判別そのものを壊した'],
], 'Sadri et al., FaMI, MICCAI 2025');
method('B15', 'SWAD', '学習中の重みを、val 損失の谷にある区間で密に平均し、平坦な解を得る', [
  ['考え方', '損失の平坦な領域の解は分布のずれに強い、という領域汎化の手法。val 損失が収束した区間から、悪化し始めるまでの重みを平均する'],
  ['本研究での使い方', 'Q2-c。n_converge 3、n_tolerance 6、r 0.3。谷の判定は epoch 単位。今回は谷の打ち切りが起きず、実質は全 epoch の重みの平均になった'],
  ['見たかったこと', 'LHCF の組み込み先。群を使わない比較相手でもある。最悪群の利得の多くは、群ではなくこの平均化から来ていた'],
], 'Cha et al., "SWAD: Domain Generalization by Seeking Flat Minima", NeurIPS 2021');
method('B16', 'JTT（Just Train Twice）', '1 回目のモデルが誤った例を重くして、2 回目を学ぶ', [
  ['考え方', '最悪群の例は ERM が誤りやすい、という見立て。1 回目の ERM で誤った例の集合を作り、その例を何倍にも重くして学び直す。群ラベルなしで最悪群が上がると報告されている'],
  ['本研究との関係', 'Q3-c の「誤りから群を推す」発想の元。実装はそのまま使わず、誤りの向きでクラスタを作る形に広げた。付録 A1 の「ERM の誤り × クラス」の区分も JTT 型'],
], 'Liu et al., "Just Train Twice: Improving Group Robustness without Training Group Information", ICML 2021');
method('B17', 'EIIL（誤りから環境を推す）', '参照モデルの誤り方から、学習を最も難しくする群（環境）を推定し、その群で学び直す', [
  ['考え方', '参照モデル（ERM）が不変な規則を最も破るように、標本を 2 つの環境に分ける。その環境で不変性を課す学習（IRM など）を回す'],
  ['本研究との関係', 'Q3-c と同じく、群ラベルなしで「規則の違う群」を誤りから推す。本研究では推した区分を hypernet の条件に使った'],
], 'Creager, Jacobsen & Zemel, "Environment Inference for Invariant Learning", ICML 2021');
method('B18', 'INLP（ラベル方向の除去）', '線形分類器で当てられる方向を見つけて射影で消すことを繰り返し、表現から特定の情報を除く', [
  ['考え方', '表現から属性 z を当てる線形分類器を学び、その重みの方向を零空間への射影で消す。当てられなくなるまで繰り返す'],
  ['本研究での使い方', 'Q2-b の「ラベル直交」の条件。DINOv2 の埋め込みからラベル（悪性か）の方向を除いてから GMM や PCA に掛けた。振り分けがラベル（難しさ）で切れるのを防ぐため'],
], 'Ravfogel et al., "Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection", ACL 2020');

pres.writeFile({ fileName: OUT }).then(() => {
  console.log(`wrote ${OUT}`);
  if (OUTLINE) { fs.writeFileSync(OUTLINE, outline.join('\n')); console.log(`wrote ${OUTLINE}`); }
});
