/* Milestone 1 explainer. All pictures and numbers come from window.M1 (docs/m1_data.js),
   which scripts/make_m1_figures.py writes from the real pipeline and the trained model. */
(function () {
  const D = window.M1;
  const $ = (s) => document.querySelector(s);
  const NS = "http://www.w3.org/2000/svg";
  const h = (tag, attrs = {}, ...kids) => {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) k === "text" ? (e.textContent = v) : e.setAttribute(k, v);
    kids.forEach((k) => e.append(k));
    return e;
  };
  const s = (tag, attrs = {}, text) => {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    if (text !== undefined) e.textContent = text;
    return e;
  };
  const tex = (id, src) => {
    const el = $(id);
    if (!el) return;
    if (window.katex) window.katex.render(src, el, { displayMode: true, throwOnError: false });
    else el.textContent = src; // offline: show the source
  };
  const num = (n) => n.toLocaleString("en-GB");
  if (!D) { document.body.prepend(h("p", { text: "m1_data.js is missing. Run: uv run python scripts/make_m1_figures.py" })); return; }

  /* ---------- header facts ---------- */
  const total = Object.values(D.params).reduce((a, b) => a + b, 0);
  const lastStep = D.loss.length ? D.loss[D.loss.length - 1][0] : 0;
  const end = (D.spend || []).filter((e) => e.event === "end").pop();
  const facts = [
    [(total / 1e6).toFixed(2) + " M", "weights in the network"],
    [String(D.vocab), "output symbols, including blank"],
    [(lastStep * 256 / 1e6).toFixed(1) + " M", "training clusters, none seen twice"],
    [D.train_fonts.length + " + " + D.held_out.length, "fonts trained on + fonts held back"],
  ];
  if (end) facts.push(["$" + end.cost.toFixed(2), "GPU rental for this run (" + Math.round(end.hours * 60) + " min)"]);
  facts.forEach(([b, t]) => $("#facts").append(h("div", { class: "fact" }, h("b", { text: b }), h("span", { text: t }))));

  /* ---------- where the network sits ---------- */
  const S = D.strip;
  $("#strip-page").src = S.page;
  $("#strip-overlay").src = S.overlay;
  $("#strip-blobs").textContent = `find ${num(S.blobs)} blobs of ink, group them into ${num(S.clusters)} clusters (a body plus its dots), order them`;
  const read = $("#strip-read");
  S.read.forEach((line, i) => {
    const truth = (S.truth[i] || "").split(" "), got = line.split(" ");
    const row = h("div");
    got.forEach((w, j) => { row.append(w === truth[j] ? w : h("mark", { text: w }), " "); });
    read.append(row);
  });
  const wrong = S.read.join(" ").split(" ").filter((w, i) => w !== S.truth.join(" ").split(" ")[i]).length;
  $("#strip-cap").textContent = `A real run on ${S.font} at ${S.px} px, a font the network never trained on. ` +
    `Top: the page as given. Middle: what the rules stage finds, with no learning. Bottom: the text read back` +
    (wrong ? `; the ${wrong} highlighted word${wrong > 1 ? "s differ" : " differs"} from what was printed.` : `, which matches what was printed exactly.`);

  /* ---------- architecture labels ---------- */
  $("#arch-out").textContent = `W/2 × ${D.vocab}`;
  $("#arch-cnn").textContent = `convolutions: ${num(D.params.cnn)} weights`;
  $("#arch-proj").textContent = `576 → 160: ${num(D.params.proj)}`;
  $("#arch-rnn").textContent = `GRU: ${num(D.params.rnn)}`;
  $("#arch-outp").textContent = `${num(D.params.out)}`;

  /* ---------- maths ---------- */
  const stack = (...lines) => String.raw`\begin{gathered}` + lines.join(String.raw` \\[6pt] `) + String.raw`\end{gathered}`;
  tex("#m-input", stack(
    String.raw`X \in [0,1]^{\,2 \times 48 \times W}`,
    String.raw`\text{rows } \big[\, b - 1.5\,s,\;\; b + 0.9\,s \,\big] \;\longrightarrow\; 48 \text{ px}`));
  tex("#m-net", stack(
    String.raw`T = \lfloor W/2 \rfloor \qquad P \in \mathbb{R}^{\,T \times ${D.vocab}}`,
    String.raw`P_{t,k} = p(\text{symbol } k \mid \text{frame } t,\, X) \qquad \textstyle\sum_k P_{t,k} = 1`));
  tex("#m-ctc", stack(
    String.raw`p(y \mid X) \;=\; \sum_{\pi \,\in\, \mathcal{B}^{-1}(y)} \; \prod_{t=1}^{T} P_{t,\pi_t}`,
    String.raw`\mathcal{L} = -\log p(y \mid X)`));
  tex("#m-train", stack(
    String.raw`\theta \;\leftarrow\; \theta - \eta_t \cdot \mathrm{AdamW}\!\big(\nabla_\theta\, \mathcal{L}\big)`,
    String.raw`\eta_{\max} = 3\times10^{-3} \qquad ${num(lastStep)} \text{ steps} \times 256 \text{ clusters}`));
  tex("#m-rules", stack(
    String.raw`\text{fits the ink}(y) \iff \mathrm{lo}(y) \;\le\; n_{\text{satellites}} \;\le\; \mathrm{hi}(y)`,
    String.raw`\text{touching}(y) \iff \exists\, i:\ \neg\,\mathrm{joins}(y_i,\, y_{i+1})`,
    String.raw`\text{lower bound used only if } \tfrac{\#\{n < \mathrm{lo}\}}{\#\text{clusters on page}} < 2\%`));

  /* ---------- cluster picker ---------- */
  const picker = $("#picker");
  D.examples.forEach((ex, i) => {
    const b = h("button", { type: "button", "aria-pressed": "false" }, h("span", { text: ex.title }), h("span", { class: "ar", lang: "ar", text: ex.label }));
    b.addEventListener("click", () => show(i));
    picker.append(b);
  });

  function show(i) {
    const ex = D.examples[i];
    [...picker.querySelectorAll("button")].forEach((b, j) => b.setAttribute("aria-pressed", String(i === j)));
    drawCrop(ex);
    drawReader(ex);
    drawCheck(ex);
  }

  /* ---------- 1. the crop ---------- */
  function drawCrop(ex) {
    const k = Math.max(3, Math.min(7, 760 / ex.W));
    const img = $("#crop-img");
    img.src = ex.img; img.width = Math.round(ex.W * k); img.height = Math.round(48 * k);
    img.alt = `The cluster ${ex.label} as cut out for the network`;
    const dims = $("#crop-dims");
    dims.replaceChildren(
      h("span", { text: `48 px tall × ${ex.W} px wide` }),
      h("span", { text: `${ex.T} frames come out (one per 2 px)` }),
      h("span", { class: "key" }, h("i", { style: "background:#151b23" }), "this cluster's ink (channel 1)"),
      h("span", { class: "key" }, h("i", { style: "background:#9d6b1f66" }), "neighbours' ink (channel 2)"),
    );
  }

  /* ---------- 3. the reader ---------- */
  function drawReader(ex) {
    const box = $("#reader");
    box.replaceChildren();
    const G = 58, k = Math.max(4, Math.min(8, 800 / ex.W)), fw = 2 * k;
    const imgH = 48 * k, rh = 22, top = imgH + 14, rows = ex.symbols.length;
    const pathY = top + rows * rh + 10, H = pathY + 30, Wpx = G + ex.W * k + 6;
    const svg = s("svg", { width: Wpx, height: H, viewBox: `0 0 ${Wpx} ${H}`, role: "img",
      "aria-label": `Frame by frame output for ${ex.label}: the network's best path collapses to ${ex.first}` });
    const fx = (t) => G + (ex.W - 2 * (t + 1)) * k; // frame 0 is at the right edge
    svg.append(s("rect", { x: G, y: 0, width: ex.W * k, height: imgH, fill: "#fff" }));
    const im = s("image", { x: G, y: 0, width: ex.W * k, height: imgH, preserveAspectRatio: "none" });
    im.setAttributeNS("http://www.w3.org/1999/xlink", "href", ex.img); im.setAttribute("href", ex.img);
    svg.append(im);
    svg.append(s("text", { x: G - 8, y: imgH / 2, "text-anchor": "end", "font-size": 11, class: "muted" }, "input"));
    ex.symbols.forEach((sym, r) => {
      const y = top + r * rh;
      svg.append(s("line", { x1: G, x2: G + ex.W * k, y1: y, y2: y, class: "grid" }));
      svg.append(s("text", r === 0
        ? { x: G - 8, y: y + 15, "text-anchor": "end", "font-size": 11, class: "muted" }
        : { x: G - 8, y: y + 17, "text-anchor": "end", "font-size": 18, class: "arabic" }, sym));
      ex.probs[r].forEach((p, t) => {
        if (p < 0.02) return;
        svg.append(s("rect", { x: fx(t) + 0.5, y: y + 1.5, width: fw - 1, height: rh - 3, rx: 1.5,
          fill: r === 0 ? "var(--muted)" : "var(--series-1)", "fill-opacity": (r === 0 ? 0.55 : 1) * p }));
      });
    });
    svg.append(s("line", { x1: G, x2: G + ex.W * k, y1: top + rows * rh, y2: top + rows * rh, class: "grid" }));
    svg.append(s("text", { x: G - 8, y: pathY + 15, "text-anchor": "end", "font-size": 11, class: "muted" }, "best"));
    ex.path.forEach((sym, t) => {
      const repeat = t > 0 && ex.path[t - 1] === sym;
      svg.append(s("text", { x: fx(t) + fw / 2, y: pathY + 16, "text-anchor": "middle",
        "font-size": sym ? Math.min(17, fw + 5) : 10, class: sym ? "arabic" : "muted",
        opacity: sym && repeat ? 0.3 : 1 }, sym || "·"));
    });
    // hover layer: one hit column per frame
    const band = s("rect", { y: 0, width: fw, height: H - 4, fill: "currentColor", "fill-opacity": 0.1, visibility: "hidden" });
    svg.append(band);
    const tip = h("div", { class: "tip", hidden: "" });
    for (let t = 0; t < ex.T; t++) {
      const hit = s("rect", { x: fx(t), y: 0, width: fw, height: H, fill: "transparent" });
      hit.addEventListener("pointerenter", () => {
        band.setAttribute("x", fx(t)); band.setAttribute("visibility", "visible");
        const ranked = ex.symbols.map((sym, r) => [sym, ex.probs[r][t]]).sort((a, b) => b[1] - a[1]).slice(0, 3).filter((x) => x[1] >= 0.005);
        tip.replaceChildren(h("b", { text: `Frame ${t + 1} of ${ex.T}` }));
        ranked.forEach(([sym, p]) => tip.append(h("div", { text: `${sym}  ${(100 * p).toFixed(1)}%` })));
        tip.hidden = false;
        const left = Math.min(fx(t) + fw + 8, Wpx - 130);
        tip.style.left = left + "px"; tip.style.top = "8px";
      });
      svg.append(hit);
    }
    svg.addEventListener("pointerleave", () => { band.setAttribute("visibility", "hidden"); tip.hidden = true; });
    box.append(svg, tip);

    // the collapse, spelled out
    const merged = ex.path.filter((sym, t) => t === 0 || ex.path[t - 1] !== sym);
    const spans = (arr, dimRepeats) => {
      const dd = h("dd", { lang: "ar" });
      arr.forEach((sym, t) => {
        const cls = !sym ? "blank" : dimRepeats && t > 0 && arr[t - 1] === sym ? "dim" : "";
        dd.append(h("span", cls ? { class: cls, text: sym || "·" } : { text: sym || "·" }), " ");
      });
      return dd;
    };
    const fin = h("dd", { class: "final", lang: "ar", text: ex.first || "(nothing)" });
    $("#collapse").replaceChildren(
      h("dt", { text: `Best symbol in each of the ${ex.T} frames` }), spans(ex.path, true),
      h("dt", { text: "Merge repeats" }), spans(merged, false),
      h("dt", { text: "Drop blanks" }), fin,
    );
  }

  /* ---------- 5. the rule check ---------- */
  const FLAG = { "missing-dots": "needs more dots than the cluster has", "unexplained-ink": "leaves satellite ink unexplained" };
  function drawCheck(ex) {
    const ok = ex.first_flags.length === 0;
    const [lo, hi] = ex.first_range;
    const right = ex.final === ex.label;
    const dl = $("#check");
    const row = (dt, ...dd) => [h("dt", { text: dt }), h("dd", {}, ...dd)];
    dl.replaceChildren(
      ...row("Network's first reading", h("span", { class: "ar", lang: "ar", text: ex.first || "(nothing)" })),
      ...row("Satellite blobs counted in the cluster", String(ex.satellites)),
      ...row("That reading needs", (lo === hi ? `${lo}` : `${lo} to ${hi}`) + " satellite blobs"),
      ...row("Rules say", h("span", { class: "verdict " + (ok ? "ok" : "no"), text: ok ? "consistent with the ink" : "flagged: " + ex.first_flags.map((f) => FLAG[f] || f).join("; ") })),
      ...row("Final reading", h("span", { class: "ar", lang: "ar", text: ex.final }), ex.final !== ex.first ? " (changed by the rules)" : ""),
      ...(ex.touching ? row("One body?", "No: these letters cannot all join, so two units have run together in this blob. The reading is kept and the join is noted.") : []),
      ...(ex.dots_fusing ? row("This page", "Dots are fusing into strokes across the page, so the too-few-dots check is switched off here.") : []),
      ...row("What was printed", h("span", { class: "ar", lang: "ar", text: ex.label }), " ", h("span", { class: "verdict " + (right ? "ok" : "no"), text: right ? "correct" : "wrong" })),
    );
    const table = $("#alts");
    table.replaceChildren(h("tr", {}, h("th", { text: "Network's readings, most likely first" }), h("th", { class: "num", text: "Probability" }), h("th", { text: "Rule check" })));
    ex.alts.forEach(([text, logp, flags]) => {
      const tr = h("tr", text === ex.final ? { class: "chosen" } : {},
        h("td", {}, h("span", { class: "ar", lang: "ar", text: text || "(nothing)" }), text === ex.final ? "  ← taken" : ""),
        h("td", { class: "num", text: (100 * Math.exp(logp)).toFixed(logp > -0.02 ? 2 : 1) + "%" }),
        h("td", { text: flags.length ? flags.map((f) => FLAG[f] || f).join("; ") : "passes" }));
      table.append(tr);
    });
  }

  /* ---------- 4. training ---------- */
  $("#loop-fonts").textContent = `HarfBuzz, ${D.train_fonts.length} fonts, 7 sizes`;
  $("#loop-note").textContent = `Held back and never trained on: ${D.held_out.join(", ")}, and the last 5% of the corpus.`;

  function lineChart(root, o) {
    const W = 440, H = 250, m = { l: 46, r: o.series.length > 1 ? 104 : 50, t: 10, b: 34 };
    const xs = o.series.flatMap((se) => se.pts.map((p) => p[0])), ys = o.series.flatMap((se) => se.pts.map((p) => p[1]));
    const x0 = 0, x1 = Math.max(...xs);
    const ty = o.log ? Math.log10 : (v) => v;
    const y0 = ty(o.yMin ?? Math.min(...ys)), y1 = ty(o.yMax ?? Math.max(...ys));
    const X = (v) => m.l + ((v - x0) / (x1 - x0)) * (W - m.l - m.r), Y = (v) => H - m.b - ((ty(v) - y0) / (y1 - y0)) * (H - m.t - m.b);
    root.append(h("h4", { text: o.title }), h("div", { class: "sub", text: o.sub }));
    if (o.series.length > 1) {
      const lg = h("div", { class: "legend" });
      o.series.forEach((se) => lg.append(h("span", {}, h("i", { style: `background:${se.color}` }), se.name)));
      root.append(lg);
    }
    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", role: "img", "aria-label": o.title + ". " + o.sub });
    o.yTicks.forEach((v) => {
      svg.append(s("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), class: "grid" }));
      svg.append(s("text", { x: m.l - 8, y: Y(v) + 4, "text-anchor": "end", "font-size": 11, class: "muted" }, o.fmt(v)));
    });
    for (let v = 0; v <= x1; v += 10000) svg.append(s("text", { x: X(v), y: H - m.b + 16, "text-anchor": "middle", "font-size": 11, class: "muted" }, v ? v / 1000 + "k" : "0"));
    svg.append(s("text", { x: (m.l + W - m.r) / 2, y: H - 2, "text-anchor": "middle", "font-size": 11, class: "muted" }, "training step"));
    o.series.forEach((se) => {
      svg.append(s("polyline", { points: se.pts.map((p) => `${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join(" "), fill: "none", stroke: se.color, "stroke-width": 2, "stroke-linejoin": "round" }));
      if (se.dots) se.pts.forEach((p) => svg.append(s("circle", { cx: X(p[0]), cy: Y(p[1]), r: 4, fill: se.color, stroke: "var(--surface)", "stroke-width": 2 })));
    });
    // direct end labels, nudged apart if they would collide
    const ends = o.series.map((se) => ({ se, p: se.pts[se.pts.length - 1], y: Y(se.pts[se.pts.length - 1][1]) })).sort((a, b) => a.y - b.y);
    for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 14) ends[i].y = ends[i - 1].y + 14;
    ends.forEach((e) => svg.append(s("text", { x: X(e.p[0]) + 8, y: e.y + 4, "font-size": 11.5, "font-weight": 600 }, o.fmt(e.p[1]) + (o.series.length > 1 ? " " + e.se.short : ""))));
    // hover: crosshair + tooltip at the nearest logged step
    const cross = s("line", { y1: m.t, y2: H - m.b, stroke: "var(--muted)", "stroke-width": 1, visibility: "hidden" });
    svg.append(cross);
    const tip = h("div", { class: "tip", hidden: "" });
    const steps = [...new Set(xs)].sort((a, b) => a - b);
    const hit = s("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent" });
    hit.addEventListener("pointermove", (ev) => {
      const r = svg.getBoundingClientRect(), vx = ((ev.clientX - r.left) / r.width) * W;
      const step = steps.reduce((a, b) => (Math.abs(X(b) - vx) < Math.abs(X(a) - vx) ? b : a));
      cross.setAttribute("x1", X(step)); cross.setAttribute("x2", X(step)); cross.setAttribute("visibility", "visible");
      tip.replaceChildren(h("b", { text: "Step " + num(step) }));
      o.series.forEach((se) => { const p = se.pts.find((q) => q[0] === step); if (p) tip.append(h("div", { text: (o.series.length > 1 ? se.name + ": " : "") + o.fmt(p[1]) })); });
      tip.hidden = false;
      tip.style.left = Math.min((X(step) / W) * r.width + 10, r.width - 150) + "px"; tip.style.top = "60px";
    });
    hit.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); tip.hidden = true; });
    svg.append(hit);
    root.append(svg, tip);
  }

  if (D.loss.length) {
    lineChart($("#chart-loss"), {
      title: "Loss falls as the network learns", sub: "CTC loss, running average, log scale. Lower is better.",
      series: [{ name: "loss", short: "", color: "var(--series-1)", pts: D.loss.map((r) => [r[0], Math.max(r[1], 0.005)]) }],
      log: true, yMin: 0.01, yMax: 10, yTicks: [0.01, 0.1, 1, 10], fmt: (v) => (v >= 1 ? v.toFixed(0) : v >= 0.1 ? v.toFixed(1) : v.toFixed(2)),
    });
  }
  if (D.evals.length) {
    const lo = Math.floor(Math.min(...D.evals.flatMap((e) => [e.seen, e.unseen])) - 0.5);
    const ticks = []; for (let v = lo; v <= 100; v += lo < 94 ? 2 : 1) ticks.push(v);
    lineChart($("#chart-acc"), {
      title: "Unseen fonts keep pace with seen fonts", sub: `Clusters read exactly right, %. The axis starts at ${lo}%, not zero.`,
      series: [
        { name: "Fonts trained on", short: "seen", color: "var(--series-1)", dots: true, pts: D.evals.map((e) => [e.step, e.seen]) },
        { name: "Fonts never seen", short: "unseen", color: "var(--series-2)", dots: true, pts: D.evals.map((e) => [e.step, e.unseen]) },
      ],
      yMin: lo, yMax: 100, yTicks: ticks, fmt: (v) => (Number.isInteger(v) ? v + "%" : v.toFixed(1) + "%"),
    });
    const t = h("table", { class: "results" }, h("tr", {}, h("th", { text: "Step" }), h("th", { class: "num", text: "Seen fonts, exact" }), h("th", { class: "num", text: "Unseen fonts, exact" }), h("th", { class: "num", text: "Seen, char. error" }), h("th", { class: "num", text: "Unseen, char. error" })));
    D.evals.forEach((e) => t.append(h("tr", {}, h("td", { text: num(e.step) }), h("td", { class: "num", text: e.seen.toFixed(2) + "%" }), h("td", { class: "num", text: e.unseen.toFixed(2) + "%" }), h("td", { class: "num", text: e.seen_cer.toFixed(2) + "%" }), h("td", { class: "num", text: e.unseen_cer.toFixed(2) + "%" }))));
    $("#acc-table").append(t);
  }

  /* ---------- results ---------- */
  const R = $("#results"), names = Object.keys(D.m1 || {});
  if (names.length) {
    R.append(h("tr", {}, h("th", { text: "Test pages" }), h("th", { text: "Rule repair" }), h("th", { class: "num", text: "Character error" }), h("th", { class: "num", text: "Word error" }), h("th", { class: "num", text: "Clusters misread" }), h("th", { class: "num", text: "Misreads the rules flagged" }), h("th", { class: "num", text: "Correct readings flagged" }), h("th", { class: "num", text: "Touching units found" })));
    names.forEach((n) => {
      const r = D.m1[n], [fonts, cond, rules] = n.split(", ");
      R.append(h("tr", {}, h("td", { text: fonts + ", " + cond }), h("td", { text: rules.replace("rules ", "") }),
        h("td", { class: "num", text: r["CER %"].toFixed(2) + "%" }), h("td", { class: "num", text: r["WER %"].toFixed(2) + "%" }),
        h("td", { class: "num", text: r["cluster errors %"].toFixed(2) + "%" }),
        h("td", { class: "num", text: rules === "rules off" ? r["errors flagged by rules %"].toFixed(0) + "%" : "–" }),
        h("td", { class: "num", text: rules === "rules off" ? r["correct clusters flagged %"].toFixed(1) + "%" : "–" }),
        h("td", { class: "num", text: rules === "rules off" && r["touching found"] !== undefined ? String(r["touching found"]) : "–" })));
    });
    const chars = D.m1[names[0]].chars;
    $("#results-cap").textContent = `Whole pages, image in, text out, compared with what was printed (roughly ${num(Math.round(chars / 1000) * 1000)} characters per row; sentences the network never trained on). "Rule repair off" takes the network's first reading; "on" lets the rules swap in a next-best reading that fits the ink. The last two columns are measured before any repair: the share of the network's misreads that the rule checks caught on their own, and the share of correct readings they flagged by mistake. "Touching units found" counts blobs whose reading shows two units have run together.`;
  } else {
    $("#results-cap").textContent = "The end-to-end page test has not been run yet.";
  }

  show(0);
})();
