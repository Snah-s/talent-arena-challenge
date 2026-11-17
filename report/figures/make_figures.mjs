import { writeFileSync } from "node:fs";
import * as d3 from "d3";
import { JSDOM } from "jsdom";

const FONT = "Liberation Sans, Arial, Helvetica, sans-serif";
const C = {
  ink: "#1f2933", muted: "#52606d", line: "#7b8794", frame: "#cbd2d9",
  home: "#2f6db3", homeBg: "#e3eefa", away: "#c8611c", awayBg: "#fcebdc",
  dead: "#6b7280", deadBg: "#eceef1", box: "#f5f7fa", train: "#e8f3ec", trainInk: "#2f7d4f",
  vision: "#f3ecfa", visionInk: "#6b46a3",
};

function canvas(width, height) {
  const dom = new JSDOM("<!doctype html><body></body>");
  const svg = d3.select(dom.window.document.body).append("svg")
    .attr("xmlns", "http://www.w3.org/2000/svg")
    .attr("width", width).attr("height", height).attr("viewBox", `0 0 ${width} ${height}`)
    .attr("font-family", FONT).attr("fill", C.ink);
  svg.append("rect").attr("width", width).attr("height", height).attr("fill", "#ffffff");
  const defs = svg.append("defs");
  for (const [id, color] of [["arrow", C.line], ["arrow-home", C.home], ["arrow-away", C.away],
                             ["arrow-dead", C.dead], ["arrow-train", C.trainInk]]) {
    defs.append("marker").attr("id", id).attr("viewBox", "0 0 10 10").attr("refX", 9).attr("refY", 5)
      .attr("markerWidth", 7).attr("markerHeight", 7).attr("orient", "auto-start-reverse")
      .append("path").attr("d", "M0,0 L10,5 L0,10 z").attr("fill", color);
  }
  return { svg, save: (name) => writeFileSync(new URL(`${name}.svg`, import.meta.url), svg.node().outerHTML + "\n") };
}

function box(g, x, y, w, h, lines, { fill = C.box, stroke = C.frame, size = 12, bold = 0, color = C.ink } = {}) {
  g.append("rect").attr("x", x).attr("y", y).attr("width", w).attr("height", h).attr("rx", 6)
    .attr("fill", fill).attr("stroke", stroke).attr("stroke-width", 1.2);
  const lh = size * 1.3, top = y + h / 2 - (lines.length - 1) * lh / 2;
  lines.forEach((t, i) => g.append("text").attr("x", x + w / 2).attr("y", top + i * lh)
    .attr("dy", "0.35em").attr("text-anchor", "middle").attr("font-size", i < bold ? size + 1 : size)
    .attr("font-weight", i < bold ? 700 : 400).attr("fill", i < bold ? color : C.ink).text(t));
  return { x, y, w, h, cx: x + w / 2, cy: y + h / 2, r: x + w, b: y + h };
}

function arrow(g, points, { color = C.line, marker = "arrow", dash = null, width = 1.5, curve = d3.curveLinear } = {}) {
  g.append("path").attr("d", d3.line().curve(curve)(points)).attr("fill", "none").attr("stroke", color)
    .attr("stroke-width", width).attr("stroke-dasharray", dash).attr("marker-end", `url(#${marker})`);
}

function label(g, x, y, t, { size = 11, anchor = "middle", color = C.muted, weight = 400, italic = false } = {}) {
  g.append("text").attr("x", x).attr("y", y).attr("dy", "0.35em").attr("text-anchor", anchor)
    .attr("font-size", size).attr("fill", color).attr("font-weight", weight)
    .attr("font-style", italic ? "italic" : null).text(t);
}

function lane(g, x, y, w, h, title, color) {
  g.append("rect").attr("x", x).attr("y", y).attr("width", w).attr("height", h).attr("rx", 10)
    .attr("fill", "none").attr("stroke", color).attr("stroke-width", 1.2).attr("stroke-dasharray", "5 4");
  label(g, x + 12, y + 14, title, { anchor: "start", color, weight: 700, size: 12 });
}

function pipeline() {
  const { svg, save } = canvas(980, 470);
  lane(svg, 10, 10, 960, 150, "TRAINING AND TUNING: external DFL bench only (CC BY 4.0)", C.trainInk);
  lane(svg, 10, 178, 960, 282, "INFERENCE: challenge game (models only run, nothing fitted)", C.home);

  const t = { fill: C.train, stroke: "#9fd0b1", bold: 1, color: C.trainInk };
  const dfl = box(svg, 28, 42, 200, 100, ["Sportec Open DFL", "7 Bundesliga matches", "TRACAB 25 Hz", "BallStatus, BallPossession"], t);
  const lab = box(svg, 262, 42, 200, 100, ["Label rule", "controlled possession:", "a switch counts after 2 s", "D where the ball is dead"], t);
  const deg = box(svg, 496, 42, 210, 100, ["Degraded copies (x2)", "ball gaps 20-60 %, noise,", "missing players, phantoms,", "pseudo Smart Tagging"], t);
  const fit = box(svg, 740, 42, 212, 100, ["Fit, leave one match out", "gradient-boosted heads", "neural owner head (GPU)", "decoder grid: alpha, durations"], t);
  arrow(svg, [[dfl.r, dfl.cy], [lab.x - 2, lab.cy]], { color: C.trainInk, marker: "arrow-train" });
  arrow(svg, [[lab.r, lab.cy], [deg.x - 2, deg.cy]], { color: C.trainInk, marker: "arrow-train" });
  arrow(svg, [[deg.r, deg.cy], [fit.x - 2, fit.cy]], { color: C.trainInk, marker: "arrow-train" });

  const inp = { bold: 1 };
  const atd = box(svg, 28, 210, 190, 62, ["Tracking (ATD)", "ball missing on 41 % of frames"], inp);
  const tag = box(svg, 28, 286, 190, 62, ["Smart Tagging", "set pieces, kick-off"], inp);
  const vid = box(svg, 28, 362, 190, 82, ["Tactical video", "WASB detector (GPU),", "only in the ATD's ball gaps"],
    { fill: C.vision, stroke: "#c9b3e6", bold: 1, color: C.visionInk });
  const fuse = box(svg, 248, 362, 170, 82, ["Ball fill", "homography + fixed rules,", "never overwrites ATD"],
    { fill: C.vision, stroke: "#c9b3e6", bold: 1, color: C.visionInk });
  const feat = box(svg, 448, 236, 160, 110, ["Per-frame features", "37 symmetric", "34 antisymmetric", "(team swap)"], { bold: 1 });
  const q = box(svg, 640, 210, 150, 58, ["q = P(alive)", "gradient boosting"], { bold: 1, fill: C.deadBg, stroke: "#b8bec6" });
  const r = box(svg, 640, 282, 150, 84, ["r = P(home | alive)", "gradient boosting", "+ neural head, w = 0.75"],
    { bold: 1, fill: C.homeBg, stroke: "#a9c6ea", color: C.home });
  const hmm = box(svg, 820, 210, 136, 156, ["4-state HMM", "live/dead", "x home/away,", "min. durations,", "Viterbi per half"], { bold: 1 });
  const csv = box(svg, 820, 392, 136, 52, ["frame,label", "H / A / D / X"], { bold: 1 });

  arrow(svg, [[atd.r, atd.cy], [feat.x - 2, feat.y + 30]]);
  arrow(svg, [[tag.r, tag.cy], [feat.x - 2, feat.cy + 10]]);
  arrow(svg, [[vid.r, vid.cy], [fuse.x - 2, fuse.cy]], { color: C.visionInk });
  arrow(svg, [[fuse.cx, fuse.y], [fuse.cx, feat.b - 20], [feat.x - 2, feat.b - 20]], { color: C.visionInk });
  arrow(svg, [[feat.r, feat.y + 30], [q.x - 2, q.cy]]);
  arrow(svg, [[feat.r, feat.b - 30], [r.x - 2, r.cy]]);
  arrow(svg, [[q.r, q.cy], [hmm.x - 2, q.cy]]);
  arrow(svg, [[r.r, r.cy], [hmm.x - 2, r.cy]]);
  arrow(svg, [[hmm.cx, hmm.b], [hmm.cx, csv.y - 2]]);

  arrow(svg, [[fit.cx - 40, fit.b], [fit.cx - 40, 190], [r.cx, 190], [q.cx, q.y - 2]],
    { color: C.trainInk, marker: "arrow-train", dash: "6 4" });
  arrow(svg, [[fit.cx + 40, fit.b], [fit.cx + 40, 190], [hmm.cx + 30, 190], [hmm.cx + 30, hmm.y - 2]],
    { color: C.trainInk, marker: "arrow-train", dash: "6 4" });
  label(svg, 820, 201, "trained heads and parameters", { color: C.trainInk, size: 10.5, italic: true });
  save("pipeline");
}

function ownerNet() {
  const { svg, save } = canvas(980, 400);
  lane(svg, 10, 10, 600, 380, "PER 5 FPS STEP (shared weights)", C.muted);
  lane(svg, 626, 10, 344, 380, "OVER A 10 s WINDOW (50 steps)", C.muted);

  const ball = box(svg, 28, 48, 150, 70, ["Ball token", "position, velocity,", "seen, time since seen"], { bold: 1 });
  const home = box(svg, 28, 140, 150, 100, ["Home players", "up to 14 tokens,", "nearest to the ball first", "7 features each"],
    { bold: 1, fill: C.homeBg, stroke: "#a9c6ea", color: C.home });
  const away = box(svg, 28, 262, 150, 100, ["Away players", "up to 14 tokens,", "nearest to the ball first", "7 features each"],
    { bold: 1, fill: C.awayBg, stroke: "#f0c29f", color: C.away });
  const emb = box(svg, 206, 120, 120, 140, ["Embedding", "2-layer MLP", "d = 64", "+ team", "embedding"], { bold: 1 });
  const att = box(svg, 354, 100, 110, 180, ["Transformer", "encoder layer", "self-attention", "over 1 + 28", "tokens,", "4 heads,", "padding mask"], { bold: 1 });
  const read = box(svg, 490, 110, 104, 160, ["Readout", "ball token", "|| mean(home)", "|| mean(away)", "= 192"], { bold: 1 });
  for (const s of [ball, home, away]) arrow(svg, [[s.r, s.cy], [emb.x - 2, Math.min(Math.max(s.cy, emb.y + 14), emb.b - 14)]]);
  arrow(svg, [[emb.r, emb.cy], [att.x - 2, att.cy]]);
  arrow(svg, [[att.r, att.cy], [read.x - 2, read.cy]]);

  const steps = 5, sx = 644, sy = 300;
  for (let i = 0; i < steps; i++) {
    const x = sx + i * 62;
    svg.append("rect").attr("x", x).attr("y", sy).attr("width", 46).attr("height", 30).attr("rx", 4)
      .attr("fill", C.box).attr("stroke", C.frame);
    label(svg, x + 23, sy + 15, i === 3 ? "..." : `t${["1", "2", "3", "", "50"][i]}`, { size: 11, color: C.ink });
  }
  arrow(svg, [[read.r, read.cy], [610, read.cy], [610, sy + 15], [sx - 2, sy + 15]]);
  const lstm = box(svg, 644, 206, 294, 60, ["Bidirectional LSTM", "hidden 64 x 2"], { bold: 1 });
  for (let i = 0; i < steps; i++) arrow(svg, [[sx + 23 + i * 62, sy], [sx + 23 + i * 62, lstm.b + 2]]);
  const head = box(svg, 644, 128, 140, 56, ["Linear", "logit f(x) per step"], { bold: 1 });
  const sym = box(svg, 800, 116, 156, 80, ["Team antisymmetry", "1/2 [f(x) - f(swap x)]", "exact by construction"],
    { bold: 1, fill: C.homeBg, stroke: "#a9c6ea", color: C.home });
  arrow(svg, [[head.cx, lstm.y], [head.cx, head.b + 2]]);
  arrow(svg, [[head.r, head.cy], [sym.x - 2, head.cy]]);
  const out = box(svg, 644, 40, 312, 60, ["r_nn = P(home | alive)", "windows overlap by 5 s; 3 seeds averaged in log-odds"],
    { bold: 1 });
  arrow(svg, [[sym.cx, sym.y], [sym.cx, out.b + 2]]);
  save("owner_nn");
}

function hmm() {
  const { svg, save } = canvas(980, 470);
  const W = 300, H = 92;
  const states = {
    LH: { x: 60, y: 60, name: "LH  live, home owns", emit: "emits H", min: ">= 3 s", c: C.home, bg: C.homeBg, st: "#a9c6ea" },
    LA: { x: 620, y: 60, name: "LA  live, away owns", emit: "emits A", min: ">= 3 s", c: C.away, bg: C.awayBg, st: "#f0c29f" },
    DH: { x: 60, y: 318, name: "DH  dead, home had it", emit: "emits D", min: ">= 4 s", c: C.dead, bg: C.deadBg, st: "#b8bec6" },
    DA: { x: 620, y: 318, name: "DA  dead, away had it", emit: "emits D", min: ">= 4 s", c: C.dead, bg: C.deadBg, st: "#b8bec6" },
  };
  for (const s of Object.values(states)) {
    svg.append("rect").attr("x", s.x).attr("y", s.y).attr("width", W).attr("height", H).attr("rx", 10)
      .attr("fill", s.bg).attr("stroke", s.st).attr("stroke-width", 1.4);
    label(svg, s.x + 12, s.y + 16, s.name, { anchor: "start", color: s.c, weight: 700, size: 13 });
    label(svg, s.x + W - 12, s.y + 16, s.emit, { anchor: "end", color: s.c, size: 12 });
    const cy = s.y + 58, xs = [s.x + 30, s.x + 74, s.x + 118, s.x + 186, s.x + 240];
    xs.forEach((x, i) => {
      if (i === 3) { label(svg, x, cy, "...", { size: 14, color: s.c }); return; }
      svg.append("circle").attr("cx", x).attr("cy", cy).attr("r", 11).attr("fill", "#ffffff")
        .attr("stroke", s.c).attr("stroke-width", 1.4);
      if (i < 4) arrow(svg, [[x + 12, cy], [xs[i + 1] - (i === 2 ? 30 : 13), cy]], { color: s.c, marker: s.c === C.home ? "arrow-home" : s.c === C.away ? "arrow-away" : "arrow-dead", width: 1.2 });
    });
    arrow(svg, [[s.x + 160, cy], [xs[4] - 13, cy]], { color: s.c, marker: s.c === C.home ? "arrow-home" : s.c === C.away ? "arrow-away" : "arrow-dead", width: 1.2 });
    const lx = xs[4];
    arrow(svg, [[lx - 6, cy - 10], [lx - 14, cy - 30], [lx + 14, cy - 30], [lx + 6, cy - 10]],
      { color: s.c, marker: s.c === C.home ? "arrow-home" : s.c === C.away ? "arrow-away" : "arrow-dead", width: 1.2, curve: d3.curveBasis });
    label(svg, s.x + 74, s.y + H - 8, `forced sub-states: ${s.min}`, { size: 10.5, color: s.c });
    label(svg, lx + 26, cy - 24, "tail", { size: 10.5, color: s.c, anchor: "start" });
  }
  const { LH, LA, DH, DA } = states;
  arrow(svg, [[LH.x + W, LH.y + 32], [LA.x - 3, LA.y + 32]], { color: C.ink });
  arrow(svg, [[LA.x, LA.y + 66], [LH.x + W + 3, LH.y + 66]], { color: C.ink });
  label(svg, 490, LH.y + 20, "p_switch: direct change of team", { color: C.ink, size: 12 });
  arrow(svg, [[LH.x + 70, LH.y + H], [DH.x + 70, DH.y - 3]], { color: C.ink });
  arrow(svg, [[DH.x + 150, DH.y], [LH.x + 150, LH.y + H + 3]], { color: C.ink });
  arrow(svg, [[LA.x + 70, LA.y + H], [DA.x + 70, DA.y - 3]], { color: C.ink });
  arrow(svg, [[DA.x + 150, DA.y], [LA.x + 150, LA.y + H + 3]], { color: C.ink });
  label(svg, LH.x + 66, 238, "1 - p_switch", { anchor: "end", size: 11.5, color: C.ink });
  label(svg, LH.x + 156, 238, "p_same", { anchor: "start", size: 11.5, color: C.ink });
  label(svg, LA.x + 66, 238, "1 - p_switch", { anchor: "end", size: 11.5, color: C.ink });
  label(svg, LA.x + 156, 238, "p_same", { anchor: "start", size: 11.5, color: C.ink });
  arrow(svg, [[DH.x + W, DH.y + 20], [LA.x + 40, LA.y + H + 3]], { color: C.muted, dash: "5 4" });
  arrow(svg, [[DA.x, DA.y + 20], [LH.x + W - 40, LH.y + H + 3]], { color: C.muted, dash: "5 4" });
  label(svg, 490, 356, "1 - p_same:", { color: C.muted, size: 11.5 });
  label(svg, 490, 374, "restart by the other team", { color: C.muted, size: 11.5 });
  label(svg, 490, 448, "Transition probabilities and duration tails from the bench labels; emissions = alpha x scaled log-likelihoods of q and r.",
    { color: C.muted, size: 11 });
  save("hmm");
}

function results() {
  const { svg, save } = canvas(980, 400);
  const m = { t: 58, b: 66, l: 60 }, Hh = 400 - m.t - m.b;
  const y = d3.scaleLinear().domain([0, 1]).range([m.t + Hh, m.t]);
  const panel = (x0, w, title) => {
    label(svg, x0, 22, title, { anchor: "start", color: C.ink, weight: 700, size: 17 });
    const g = svg.append("g");
    for (const v of [0, 0.2, 0.4, 0.6, 0.8, 1.0]) {
      g.append("line").attr("x1", x0).attr("x2", x0 + w).attr("y1", y(v)).attr("y2", y(v))
        .attr("stroke", v === 0 ? C.line : "#e4e7eb");
      label(g, x0 - 10, y(v), v.toFixed(1), { anchor: "end", size: 14 });
    }
    return g;
  };
  const bench = [["baseline", 0.182, C.dead], ["v1", 0.544, "#8fb3df"], ["v6", 0.605, "#5a8fd0"], ["v7", 0.655, C.home]];
  const x1 = d3.scaleBand().domain(bench.map((d) => d[0])).range([m.l + 10, 470]).padding(0.35);
  const g1 = panel(m.l + 10, 460 - m.l, "DFL bench, leave one match out (spellF1)");
  for (const [k, v, c] of bench) {
    g1.append("rect").attr("x", x1(k)).attr("y", y(v)).attr("width", x1.bandwidth()).attr("height", y(0) - y(v)).attr("fill", c).attr("rx", 2);
    label(g1, x1(k) + x1.bandwidth() / 2, y(v) - 13, v.toFixed(3), { color: C.ink, size: 14.5 });
    label(g1, x1(k) + x1.bandwidth() / 2, y(0) + 18, k, { color: C.ink, size: 15 });
  }
  g1.append("line").attr("x1", m.l + 10).attr("x2", 470).attr("y1", y(0.79)).attr("y2", y(0.79))
    .attr("stroke", C.trainInk).attr("stroke-dasharray", "6 4").attr("stroke-width", 1.4);
  label(g1, 470, y(0.79) - 13, "v6 with a perfect owner: 0.79", { anchor: "end", color: C.trainInk, size: 14.5 });
  label(g1, (m.l + 480) / 2, 400 - 16, "v8 adds the video ball fill: not measurable on the bench (no video)", { size: 13.5, italic: true });

  const pub = [["v6", 0.645, 0.44, 0.83, "#5a8fd0"], ["v7", 0.650, 0.45, 0.84, C.home], ["v8 (submitted)", 0.671, 0.46, 0.86, "#1d4f8f"]];
  const x2 = d3.scaleBand().domain(pub.map((d) => d[0])).range([570, 960]).padding(0.4);
  const g2 = panel(570, 390, "Public 5-min window (spellF1, 5-95 % interval)");
  for (const [k, v, lo, hi, c] of pub) {
    const cx = x2(k) + x2.bandwidth() / 2;
    g2.append("rect").attr("x", x2(k)).attr("y", y(v)).attr("width", x2.bandwidth()).attr("height", y(0) - y(v)).attr("fill", c).attr("rx", 2).attr("opacity", 0.85);
    g2.append("line").attr("x1", cx).attr("x2", cx).attr("y1", y(lo)).attr("y2", y(hi)).attr("stroke", C.ink).attr("stroke-width", 1.4);
    for (const e of [lo, hi]) g2.append("line").attr("x1", cx - 8).attr("x2", cx + 8).attr("y1", y(e)).attr("y2", y(e)).attr("stroke", C.ink).attr("stroke-width", 1.4);
    label(g2, cx, y(hi) - 15, v.toFixed(3), { color: C.ink, size: 14.5 });
    label(g2, cx, y(0) + 18, k, { color: C.ink, size: 15 });
  }
  label(g2, 765, 400 - 16, "17 possessions: the intervals overlap almost entirely", { size: 13.5, italic: true });
  save("results");
}

pipeline();
ownerNet();
hmm();
results();
