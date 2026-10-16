// 研究進捗報告スライド slide-label-semantics.pptx を生成する（AUROC の年齢差とラベルの付け方）。
// 先に figures.py で figures/ に図を書き出してから実行する:
//   uv run python docs/slides/label-semantics/figures.py
//   NODE_PATH=<pptxgenjs のある node_modules> node docs/slides/label-semantics/build.js [出力 pptx]
// 数値は analysis/archive/direction_poc/results/d5/{metrics,decomposition,bootstrap}.csv から読む。
// テーマは docs/slides/chexpert-q1/build.js と同じ。
const fs = require('fs');
const path = require('path');
const pptxgen = require('pptxgenjs');
const pres = new pptxgen();
pres.layout = 'LAYOUT_16x9'; // 10 x 5.625
const F = 'Yu Gothic';
const INK = '222222', MUTED = '666666', RULE = 'BBBBBB';
const THEME = '1F3B57';
const DECK = 'CheXpert No Finding の年齢差とラベルの付け方：研究進捗';
const OUT = process.argv[2] || path.join(__dirname, 'slide-label-semantics.pptx');
const FIG = path.join(__dirname, 'figures');

let page = 0;
function frame(s) {
  s.background = { color: 'FFFFFF' }; page++;
  s.addShape(pres.shapes.LINE, { x: 0.5, y: 5.2, w: 9, h: 0, line: { color: RULE, width: 0.5 } });
  s.addText(DECK, { x: 0.5, y: 5.25, w: 7, h: 0.25, fontFace: F, fontSize: 8, color: MUTED, margin: 0 });
  s.addText(String(page), { x: 9.0, y: 5.25, w: 0.5, h: 0.25, fontFace: F, fontSize: 8, color: MUTED, align: 'right', margin: 0 });
}
function slide(title, crumb, message) {
  const s = pres.addSlide(); frame(s);
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.7, fill: { color: THEME }, line: { color: THEME } });
  s.addText(title, { x: 0.5, y: 0, w: 6.6, h: 0.7, fontFace: F, fontSize: 18, bold: true, color: 'FFFFFF', valign: 'middle', margin: 0 });
  if (crumb) s.addText(crumb, { x: 7.1, y: 0, w: 2.4, h: 0.7, fontFace: F, fontSize: 9, color: 'C9D3DC', align: 'right', valign: 'middle', margin: 0 });
  s.addText(message, { x: 0.5, y: 0.8, w: 9, h: 0.62, fontFace: F, fontSize: 14, bold: true, color: THEME, valign: 'middle', margin: 0 });
  s.addShape(pres.shapes.LINE, { x: 0.5, y: 1.48, w: 9, h: 0, line: { color: THEME, width: 0.75 } });
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
}
function figure(s, name, ratio, { x = 5.1, y = 1.6, w = 4.4 } = {}) {
  let h = w / ratio;
  if (h > 3.25) { h = 3.25; w = h * ratio; x = 9.5 - w; }
  s.addImage({ path: path.join(FIG, `${name}.png`), x, y, w, h });
}
function refs(s, text) {
  s.addText(text, { x: 0.5, y: 4.88, w: 9, h: 0.28, fontFace: F, fontSize: 7.5, color: MUTED, valign: 'bottom', margin: 0 });
}

// 結果の CSV を読む（値にカンマを含まない前提）
function readCsv(name) {
  const [h, ...ls] = fs.readFileSync(path.join(__dirname, '../../../analysis/archive/direction_poc/results/d5', name), 'utf8').trim().split('\n');
  const c = h.split(',');
  return ls.map((l) => Object.fromEntries(l.split(',').map((v, i) => [c[i], v])));
}
function find(rows, pred, what) {
  const r = rows.find(pred);
  if (!r) throw new Error(`${what} が結果の CSV に無い`);
  return r;
}
function quantile(xs, q) {
  const s = [...xs].sort((a, b) => a - b), i = (s.length - 1) * q, lo = Math.floor(i);
  return s[lo] + (s[Math.ceil(i)] - s[lo]) * (i - lo);
}
const BASE = 'M01 ResNet ERM';
const METRICS = readCsv('metrics.csv');
const DEC = find(readCsv('decomposition.csv'), (r) => r.model === BASE && r.band === '<65 vs ≥65', '分解');
const BOOT = readCsv('bootstrap.csv').filter((r) => r.model === BASE);
const gap = (model, pos) => Number(find(METRICS, (r) => r.model === model && r.positives === pos, `${model}/${pos}`).age_gap);
const f3 = (v) => Number(v).toFixed(3);
const sg = (v) => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(3);
const pct = (v) => `${Math.round(Number(v) * 100)}%`;
const ci = (col, fmt = f3) => {
  const xs = BOOT.map((r) => Number(r[col]));
  return `[${fmt(quantile(xs, 0.025))}, ${fmt(quantile(xs, 0.975))}]`;
};
const CONDITIONS = ['M01 ResNet ERM', 'M02 ResNet DRO', 'M03 Spatial ERM', 'M04 Spatial DRO', 'M06 Spatial 2-stage DRO', 'M07 HyperAdapt ERM'];
const range = (pos) => {
  const v = CONDITIONS.map((m) => gap(m, pos));
  return `${f3(Math.min(...v))}〜${f3(Math.max(...v))}`;
};
const share = (dec) => Math.round((Number(DEC[dec]) / Number(DEC.gap)) * 10);

// 1. 2 種類のレポート
{
  const s = slide('No Finding の陽性に混ざる 2 種類のレポート', 'ラベルの付け方',
    '高齢の陽性ほど、追跡撮影の質問に答えただけのレポートが多い');
  bullets(s, [
    ['No Finding の陽性は「レポートが異常に触れていない」画像'],
    ['レポートの文面から、陽性を種別に分けた'],
    ['正常を明言：画像全体を見て、正常と書いたもの', 1],
    ['追跡撮影の質問に答えただけ：デバイスの位置・気胸の有無・前回との比較などに答え、ほかの所見に触れないもの', 1],
    ['後者は、レポートが触れない異常が画像に残っていても陽性になる'],
    [`高齢（≥65）の陽性では、追跡撮影型が ${pct(DEC['w old focused_followup'])}（若年 ${pct(DEC['w young focused_followup'])}）、正常を明言が ${pct(DEC['w old affirmed_normal'])}（若年 ${pct(DEC['w young affirmed_normal'])}）`],
  ]);
  figure(s, 'composition', 4.6 / 2.6, { y: 1.75 });
  refs(s, 'CheXpert test 19,617 枚（陽性は No Finding）。種別はレポートの正規表現による分類（data_pipeline の label_construct.py）');
}

// 2. 絞ると年齢差が縮む
{
  const s = slide('正常を明言した陽性で測ると年齢差は縮む', 'ラベルの付け方',
    `陽性を正常を明言したものに絞ると、AUROC の年齢差は ${f3(gap(BASE, 'all'))} から ${f3(gap(BASE, 'affirmed_normal'))} に減る`);
  const diff = BOOT.map((r) => Number(r['diff age_gap']));
  bullets(s, [
    ['学習はし直さず、既存の予測を評価し直した'],
    ['陰性は全件残し、陽性だけを種別で絞る', 1],
    [`年齢差（<65 − ≥65）：${f3(gap(BASE, 'all'))} → ${f3(gap(BASE, 'affirmed_normal'))}`],
    [`縮み幅 ${sg(gap(BASE, 'affirmed_normal') - gap(BASE, 'all'))} [${sg(quantile(diff, 0.025))}, ${sg(quantile(diff, 0.975))}]`, 1],
    [`絞った後の区間 ${ci('aff age_gap')} は 0 を含む`, 1],
    [`追跡撮影型だけの陽性では、年齢差 ${f3(gap(BASE, 'focused_followup'))} が残る`],
    [`年齢差の内訳：種別の構成の違いで約 ${share('composition')} 割、追跡撮影型の中の差で約 ${share('within focused_followup')} 割`],
  ]);
  figure(s, 'auroc_by_age', 4.6 / 3.0);
  refs(s, 'ResNet-50 ERM（race-known 主比較、3 seed の logit 平均）。区間は患者単位 bootstrap 500 回の 95%。≥65 の正常を明言した陽性は 140 件');
}

// 3. 6 条件がそろう
{
  const s = slide('既存の 6 条件でも同じ', 'ラベルの付け方',
    `どの条件も年齢差が ${range('affirmed_normal')} にそろう。手法の差もラベルの差に埋もれていた`);
  bullets(s, [
    ['race-known の主比較 6 条件で、同じ測り直しをした'],
    [`陽性全体：年齢差 ${range('all')}`],
    [`正常を明言した陽性：${range('affirmed_normal')}`],
    ['構成（ResNet・Spatial LoRA・HyperAdapt）や損失（ERM・GroupDRO）の違いは、どちらで測っても小さい'],
    ['最悪群 AUROC などで手法を比べた差は、主にラベルの付け方の差を見ていた可能性が高い'],
  ]);
  figure(s, 'six_conditions', 4.6 / 2.75);
  refs(s, '各条件 3 seed の logit 平均。test は CheXpert 19,617 枚で共通');
}

// 4. ほかのデータセット（analysis/archive/direction_poc/d5_other_datasets.py の出力）
{
  const OTHER = readCsv('other_datasets.csv');
  const row = (ds, axis) => find(OTHER, (r) => r.dataset === ds && r.axis === axis, `${ds}/${axis}`);
  const cell = (r, m) => {
    const sig = Number(r[`${m} gap_lo`]) > 0 || Number(r[`${m} gap_hi`]) < 0;
    return { text: `${sg(Number(r[`${m} gap`]))}\n[${sg(Number(r[`${m} gap_lo`]))}, ${sg(Number(r[`${m} gap_hi`]))}]`,
      options: sig ? { bold: true, color: THEME } : {} };
  };
  const ROWS = [
    ['HAM10000', 'age_group_65', '年齢（<65 − ≥65）'],
    ['HAM10000', 'sex', '性別（女性 − 男性）'],
    ['FairVLMed', 'age_group_65', '年齢（<65 − ≥65）'],
    ['FairVLMed', 'race', 'race（White − Black）'],
    ['FairVLMed', 'gender', '性別（女性 − 男性）'],
  ];
  const s = slide('ほかのデータセットの属性差', 'ラベルの付け方',
    '年齢の差は、どのデータでも順位より動作点のずれのほうが大きい');
  const H = { bold: true, color: 'FFFFFF', fill: { color: THEME } };
  s.addTable([
    ['データ', '属性（差の向き）', 'AUROC の差', 'TPR の差', 'FPR の差'].map((t) => ({ text: t, options: H })),
    ...ROWS.map(([ds, axis, label], i) => {
      const r = row(ds, axis), fill = i % 2 ? { fill: { color: 'E8EEF3' } } : {};
      return [{ text: ds }, { text: label }, cell(r, 'auroc'), cell(r, 'tpr'), cell(r, 'fpr')]
        .map((c) => ({ text: c.text, options: { ...fill, ...(c.options || {}) } }));
    }),
  ], { x: 0.5, y: 1.6, w: 9, colW: [1.3, 2.2, 1.85, 1.85, 1.8], fontFace: F, fontSize: 8.5, color: INK,
    border: { type: 'solid', pt: 0.5, color: RULE }, valign: 'middle', align: 'center', margin: [1, 4, 1, 4] });
  const ha = row('HAM10000', 'age_group_65'), fa = row('FairVLMed', 'age_group_65'), fg = row('FairVLMed', 'gender');
  bullets(s, [
    [`HAM10000：AUROC の差は年齢・性別とも 0 と区別できない。高齢は FPR が ${f3(-Number(ha['fpr gap']))} 高い（得点を一様に上げる）`],
    [`FairVLMed の年齢：高齢のほうが AUROC は高い。TPR・FPR は高齢で ${f3(-Number(fa['tpr gap']))}・${f3(-Number(fa['fpr gap']))} 高く、順位の差の約 5 倍`],
    [`FairVLMed の性別：AUROC の差 ${f3(-Number(fg['auroc gap']))} が残り、FPR の差と同じ程度。ラベルの付け方で説明できるかは未確認`],
  ], { y: 3.75, w: 9, h: 1.1, size: 10 });
  refs(s, 'ResNet-50 ERM（3 seed の logit 平均）、しきい値 0.5。太字は 95% 区間が 0 を含まない（HAM は病変単位、FairVLMed は画像単位の bootstrap）');
}

pres.writeFile({ fileName: OUT }).then((f) => console.log(`wrote ${f}`));
