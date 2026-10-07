"use strict";
const evidence = JSON.parse(document.getElementById("evidence").textContent);
const content = document.getElementById("content");
const colors = ["#087b94", "#d48216", "#a54e8b", "#4a9161", "#a35b4c", "#5c70a4", "#bd6b94", "#899442", "#886040", "#538a9c"];
const modelNames = {
  ols: "Ordinary least squares",
  ridge: "Ridge",
  bayesian_ridge: "Bayesian ridge",
  logistic: "Logistic regression",
  lda: "LDA",
  qda: "QDA",
  knn: "KNN",
  naive_bayes: "Gaussian Naive Bayes",
  tree: "Decision tree",
  bagging: "Bagging",
  random_forest: "Random Forest",
  extra_trees: "Extra Trees",
  adaboost: "AdaBoost",
  hist_boosting: "Histogram boosting",
  catboost: "CatBoost",
  linear_svm: "Linear SVM",
  rbf_svm: "RBF SVM",
  rbf_svr: "RBF SVR",
  voting: "Voting",
  stacking: "Stacking",
  l1: "Direct predictor",
  gan: "Original hinge GAN",
  cvae: "Conditional VAE",
  diffusion: "Conditional diffusion",
  ddcolor: "DDColor",
  ct2: "CT2",
  bigcolor: "BigColor",
  swinir: "SwinIR",
  realesrgan: "Real-ESRGAN",
  direct_finetune: "Direct fine-tuning control",
  hinge_sn: "Hinge + spectral normalization",
  non_saturating_r1: "Non-saturating + SN + R1",
  least_squares: "Least-squares + SN",
  wasserstein_gp: "Wasserstein + gradient penalty"
};

function name(value) {
  if (value.endsWith("_final_epoch")) return name(value.slice(0, -12)) + " · fixed epoch 20";
  return modelNames[value] || value.replaceAll("_", " ");
}

function el(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function fmt(value, digits = 3) {
  return value === null || value === undefined ? "—" : Number(value).toFixed(digits);
}

function mean(values) {
  return values.reduce((a, b) => a + b, 0) / values.length;
}

function sd(values) {
  return values.length < 2 ? 0 : Math.sqrt(values.reduce((a, b) => a + (b - mean(values)) ** 2, 0) / (values.length - 1));
}

function note(text) {
  content.append(el("p", text, "note"));
}

function panel(title) {
  const box = el("div", undefined, "panel");
  if (title) box.append(el("h2", title));
  return box;
}

function metrics(items) {
  const box = el("div", undefined, "metrics");
  items.forEach(([label, value]) => {
    const card = el("div", undefined, "metric");
    card.append(el("strong", String(value)), el("span", label));
    box.append(card);
  });
  content.append(box);
}

function select(title, values, selected) {
  const label = el("label", title);
  const input = el("select");
  input.dataset.control = title;
  values.forEach(value => {
    const option = el("option", name(String(value)));
    option.value = String(value);
    input.append(option);
  });
  if (selected !== undefined) input.value = String(selected);
  label.append(input);
  return [label, input];
}

function slider(title, min, max, value, step = 1) {
  const label = el("label", title);
  const input = el("input");
  input.type = "range";
  Object.assign(input, {
    min,
    max,
    step
  });
  input.value = String(value);
  input.dataset.control = title;
  const output = el("span", String(value));
  input.addEventListener("input", () => output.textContent = input.value);
  label.append(input, output);
  return [label, input];
}

function table(headers, rows) {
  const wrap = el("div", undefined, "table-scroll");
  const node = el("table");
  const head = el("thead");
  const hr = el("tr");
  headers.forEach(h => hr.append(el("th", h)));
  head.append(hr);
  const body = el("tbody");
  rows.forEach(row => {
    const tr = el("tr");
    row.forEach(value => tr.append(el("td", String(value))));
    body.append(tr);
  });
  node.append(head, body);
  wrap.append(node);
  return wrap;
}

function link(path, text = "Source metrics") {
  const a = el("a", text);
  a.href = path;
  return a;
}

function aggregate(rows, key, group = "model") {
  return [...new Set(rows.map(r => r[group]))].map(method => {
    const fits = rows.filter(r => r[group] === method);
    return {
      method,
      fits,
      mean: mean(fits.map(r => r[key])),
      sd: sd(fits.map(r => r[key]))
    };
  });
}

function svgNode(tag, attributes, text) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  Object.entries(attributes || {}).forEach(([k, v]) => node.setAttribute(k, v));
  if (text !== undefined) node.textContent = text;
  return node;
}

function plot(series, {
  xlabel = "",
  ylabel = "",
  points = false,
  logY = false,
  intervals = false
} = {}) {
  const svg = svgNode("svg", {
    viewBox: "0 0 740 360",
    class: "plot",
    role: "img",
    "aria-label": `${ylabel} versus ${xlabel}`
  });
  const all = series.flatMap(s => s.points);
  if (!all.length) return svg;
  const yTransform = y => logY ? Math.log10(Math.max(y, 1e-12)) : y;
  let xmin = Math.min(...all.map(p => p.x)),
    xmax = Math.max(...all.map(p => p.x));
  let ymin = Math.min(...all.map(p => yTransform(intervals ? p.lo ?? p.y : p.y))),
    ymax = Math.max(...all.map(p => yTransform(intervals ? p.hi ?? p.y : p.y)));
  if (xmax === xmin) xmax = xmin + 1;
  if (ymax === ymin) ymax = ymin + 1;
  const pad = (ymax - ymin) * .05;
  ymin -= pad;
  ymax += pad;
  const px = x => 60 + 640 * (x - xmin) / (xmax - xmin),
    py = y => 300 - 260 * (yTransform(y) - ymin) / (ymax - ymin);
  for (let i = 0; i <= 4; i++) {
    const x = xmin + (xmax - xmin) * i / 4,
      y = ymin + (ymax - ymin) * i / 4;
    svg.append(svgNode("line", {
      x1: 60,
      x2: 700,
      y1: 300 - 260 * i / 4,
      y2: 300 - 260 * i / 4,
      stroke: "#dbe6ed"
    }));
    svg.append(svgNode("text", {
      x: 4,
      y: 304 - 260 * i / 4
    }, fmt(logY ? 10 ** y : y, logY ? 3 : 2)));
    svg.append(svgNode("text", {
      x: px(x) - 12,
      y: 325
    }, fmt(x, 1)));
  }
  series.forEach((s, index) => {
    const color = s.color || colors[index % colors.length];
    if (points) {
      s.points.forEach(p => {
        if (intervals && p.lo !== undefined) svg.append(svgNode("line", {
          x1: px(p.x),
          x2: px(p.x),
          y1: py(p.lo),
          y2: py(p.hi),
          stroke: color,
          "stroke-opacity": .12
        }));
        const dot = svgNode("circle", {
          cx: px(p.x),
          cy: py(p.y),
          r: p.radius || 2.5,
          fill: p.color || color,
          "fill-opacity": .65
        });
        dot.append(svgNode("title", {}, `${s.label}: x=${fmt(p.x)}, y=${fmt(p.y)}`));
        svg.append(dot);
      });
    } else {
      const line = svgNode("path", {
        d: s.points.map((p, i) => `${i?"L":"M"}${px(p.x)},${py(p.y)}`).join(" "),
        stroke: color,
        "stroke-width": 2.5,
        fill: "none"
      });
      line.append(svgNode("title", {}, s.label));
      svg.append(line);
    }
  });
  svg.append(svgNode("text", {
    x: 320,
    y: 355
  }, xlabel), svgNode("text", {
    x: 60,
    y: 20
  }, ylabel + (logY ? " (log scale)" : "")));
  return svg;
}

function legend(series) {
  const box = el("div", undefined, "legend");
  series.forEach((s, i) => {
    const label = el("span", s.label);
    label.style.setProperty("--color", s.color || colors[i % colors.length]);
    box.append(label);
  });
  return box;
}

function heading(title, text) {
  content.replaceChildren(el("h2", title), el("p", text, "muted"));
}

function classification() {
  heading("Classification and calibrated decisions", "Explore classical model families on identical data roles. The tables summarize all three fitted seeds; the detailed panel shows one fitted model.");
  const controls = el("div", undefined, "controls");
  const [dl, dataset] = select("Dataset", [...new Set(evidence.classification.map(r => r.dataset))], "adult");
  const [ml, model] = select("Model", []);
  const [sl, seed] = select("Fitting seed", [0, 1, 2]);
  const [tl, threshold] = slider("Exploratory test threshold", 0, 1, .5, .01);
  controls.append(dl, ml, sl, tl);
  content.append(controls);
  const results = el("div");
  content.append(results);

  function render() {
    results.replaceChildren();
    const row = evidence.classification.find(r => r.dataset === dataset.value && r.model === model.value && r.seed === Number(seed.value));
    const binary = row.thresholds.length > 0;
    threshold.disabled = !binary;
    const kpi = el("div", undefined, "metrics");
    [
      ["Default decision accuracy", `${fmt(100*row.accuracy,2)}%`],
      ["Calibrated NLL", fmt(row.log_loss)],
      ["Prediction-set coverage", `${fmt(100*row.set_coverage,1)}%`],
      ["Test examples", row.n.toLocaleString()]
    ].forEach(([a, b]) => {
      const box = el("div", undefined, "metric");
      box.append(el("strong", b), el("span", a));
      kpi.append(box);
    });
    results.append(kpi);
    const grid = el("div", undefined, "grid"),
      cal = panel("Confidence calibration"),
      conf = panel("Confusion and threshold");
    const calibrationSeries = [{
      label: "Observed accuracy",
      points: row.reliability
    }, {
      label: "Ideal calibration",
      points: [{
        x: 0,
        y: 0
      }, {
        x: 1,
        y: 1
      }],
      color: "#8999a4"
    }];
    cal.append(plot(calibrationSeries, {
      xlabel: "Mean top-class confidence",
      ylabel: "Observed accuracy"
    }), legend(calibrationSeries));
    if (binary) {
      const r = row.thresholds[Math.round(Number(threshold.value) * 100)];
      conf.append(table(["Actual / predicted", "Class0", "Class1"], [
        ["Class0", r.tn, r.fp],
        ["Class1", r.fn, r.tp]
      ]));
      const precision = r.tp / (r.tp + r.fp || 1),
        recall = r.tp / (r.tp + r.fn || 1);
      conf.append(el("p", `Threshold ${fmt(r.threshold,2)} · precision ${fmt(precision)} · recall ${fmt(recall)} · accuracy ${fmt((r.tp+r.tn)/row.n)}.`), el("p", dataset.value === "breast_cancer" ? "Class1 denotes benign in this educational dataset." : "Class1 denotes income above $50,000."));
    } else conf.append(table(["Actual / predicted", ...row.confusion.map((_, i) => String(i))], row.confusion.map((r, i) => [String(i), ...r])));
    conf.append(el("p", "The threshold view is descriptive on the fixed test set. It does not select a new deployment threshold.", "muted"), link(row.metrics));
    grid.append(cal, conf);
    results.append(grid);
    const comparison = panel("Compare all fitted families");
    const all = evidence.classification.filter(r => r.dataset === dataset.value);
    comparison.append(table(["Model", "Fits", "Mean accuracy", "Mean NLL ± fit SD", "Mean macro-F1", "Mean coverage", "Mean selection seconds"], aggregate(all, "log_loss").sort((a, b) => a.mean - b.mean).map(r => [name(r.method), r.fits.length, `${fmt(100*mean(r.fits.map(f=>f.accuracy)),2)}%`, `${fmt(r.mean)} ± ${fmt(r.sd)}`, fmt(mean(r.fits.map(f => f.macro_f1))), `${fmt(100*mean(r.fits.map(f=>f.set_coverage)),1)}%`, fmt(mean(r.fits.map(f => f.seconds_including_selection)), 1)])));
    results.append(comparison);
  }

  function methods() {
    ml.querySelector("select").replaceChildren();
    [...new Set(evidence.classification.filter(r => r.dataset === dataset.value).map(r => r.model))].forEach(m => {
      const option = el("option", name(m));
      option.value = m;
      model.append(option);
    });
    model.value = "hist_boosting";
    render();
  }
  dataset.addEventListener("change", methods);
  [model, seed, threshold].forEach(node => node.addEventListener(node === threshold ? "input" : "change", render));
  methods();
  note("Macro-F1 selects presets on validation. Probability calibration and conformal calibration are separate. Full Adult excludes RBF SVM; numeric datasets and the kernel learning-curve study cover it. Candidate counts and compute differ by family.");
}

function regression() {
  heading("Regression, intervals and geographic shift", "The same model family can behave differently when deployment geography changes. Inspect prediction error and observed interval coverage together.");
  const controls = el("div", undefined, "controls");
  const [pl, protocol] = select("Test protocol", ["random", "geographic"]);
  const [ml, model] = select("Model", [...new Set(evidence.regression.map(r => r.model))], "hist_boosting");
  const [sl, seed] = select("Fitting seed", [0, 1, 2]);
  controls.append(pl, ml, sl);
  content.append(controls);
  const results = el("div");
  content.append(results);

  function render() {
    results.replaceChildren();
    const row = evidence.regression.find(r => r.protocol === protocol.value && r.model === model.value && r.seed === Number(seed.value));
    const top = el("div", undefined, "metrics");
    [
      ["Test RMSE", fmt(row.rmse)],
      ["Test MAE", fmt(row.mae)],
      ["90% interval coverage", `${fmt(100*row.interval_coverage,1)}%`],
      ["Mean interval width", fmt(row.interval_width)]
    ].forEach(([a, b]) => {
      const box = el("div", undefined, "metric");
      box.append(el("strong", b), el("span", a));
      top.append(box);
    });
    results.append(top);
    const box = panel("Predictions and calibrated intervals");
    box.append(plot([{
      label: "Prediction ± conformal interval",
      points: row.points
    }, {
      label: "Ideal prediction",
      points: [{
        x: 0,
        y: 0,
        lo: 0,
        hi: 0
      }, {
        x: 5,
        y: 5,
        lo: 5,
        hi: 5
      }],
      color: "#a54e8b"
    }], {
      points: true,
      intervals: true,
      xlabel: "Observed house value ($100,000)",
      ylabel: "Predicted house value ($100,000)"
    }), el("p", `First 200 predetermined test rows shown; summary metrics use all ${row.n.toLocaleString()} examples.`, "muted"), link(row.metrics));
    results.append(box);
    const comparison = panel("Compare all regression families");
    comparison.append(table(["Model", "Mean RMSE ± fit SD", "Mean MAE", "Mean coverage", "Mean width"], aggregate(evidence.regression.filter(r => r.protocol === protocol.value), "rmse").sort((a, b) => a.mean - b.mean).map(r => [name(r.method), `${fmt(r.mean)} ± ${fmt(r.sd)}`, fmt(mean(r.fits.map(f => f.mae))), `${fmt(100*mean(r.fits.map(f=>f.interval_coverage)),1)}%`, fmt(mean(r.fits.map(f => f.interval_width)))])));
    results.append(comparison);
  }
  [protocol, model, seed].forEach(node => node.addEventListener("change", render));
  render();
  note("Fit, validation and calibration roles are disjoint. Geographic test data violate the exchangeability premise behind ordinary conformal coverage. Models are frozen before evaluating the test. Intervals target marginal coverage, not certainty for each house.");
}

function clustering() {
  heading("Unsupervised structure and noise coverage", "Compare centroid, mixture, graph and adaptive-density methods. A method can find clean-looking clusters while leaving many observations unassigned.");
  const controls = el("div", undefined, "controls");
  const [dl, dataset] = select("Dataset", Object.keys(evidence.clustering.data));
  const [ml, method] = select("Method", [...new Set(evidence.clustering.rows.map(r => r.method))], "hdbscan");
  const [sl, seed] = select("Fitting seed", [0, 1, 2]);
  const [vl, view] = select("Color by", ["learned_cluster", "reference_label"]);
  controls.append(dl, ml, sl, vl);
  content.append(controls);
  const results = el("div");
  content.append(results);

  function render() {
    results.replaceChildren();
    const row = evidence.clustering.rows.find(r => r.case === dataset.value && r.method === method.value && r.seed === Number(seed.value));
    const data = evidence.clustering.data[dataset.value];
    const labels = view.value === "reference_label" ? data.true_labels : data[`${method.value}_${seed.value}`];
    const kpis = el("div", undefined, "metrics");
    [
      ["ARI on every point", fmt(row.ari_all_points)],
      ["Assigned observations", `${fmt(100*row.nonnoise_fraction,1)}%`],
      ["Discovered clusters", row.clusters],
      ["Non-noise silhouette", fmt(row.silhouette_nonnoise)]
    ].forEach(([a, b]) => {
      const box = el("div", undefined, "metric");
      box.append(el("strong", b), el("span", a));
      kpis.append(box);
    });
    results.append(kpis);
    const box = panel("Geometry of the selected result");
    box.append(plot([{
      label: "Observation",
      points: data.projection.slice(0, 1200).map((p, i) => ({
        x: p[0],
        y: p[1],
        color: labels[i] < 0 ? "#9ba8b2" : colors[labels[i] % colors.length]
      }))
    }], {
      points: true,
      xlabel: "PCA coordinate1 (display)",
      ylabel: "PCA coordinate2 (display)"
    }), el("p", `Gray marks noise when colored by learned cluster. First ${Math.min(data.projection.length,1200)} fixed-order observations shown. Clustering uses the declared full representation, not only these display coordinates.`, "muted"));
    results.append(box);
    const comparison = panel("Cluster agreement and coverage");
    comparison.append(table(["Method", "Mean ARI ± fit SD", "Mean assigned fraction", "Mean clusters"], aggregate(evidence.clustering.rows.filter(r => r.case === dataset.value), "ari_all_points", "method").sort((a, b) => b.mean - a.mean).map(r => [name(r.method), `${fmt(r.mean)} ± ${fmt(r.sd)}`, `${fmt(100*mean(r.fits.map(f=>f.nonnoise_fraction)),1)}%`, fmt(mean(r.fits.map(f => f.clusters)), 1)])));
    results.append(comparison);
  }
  [dataset, method, seed, view].forEach(node => node.addEventListener("change", render));
  render();
  note("Transductive unsupervised study: scaling/PCA and clustering use all unlabeled observations. Class labels are reserved for diagnostics and optional display. Count-constrained methods receive a declared cluster-count prior; DBSCAN/HDBSCAN determine clusters and noise. ARI includes noise observations. These are not held-out classification scores.");
}

function legendre(x, degree = 5) {
  const values = [1, x];
  for (let n = 2; n <= degree; n++) values.push(((2 * n - 1) * x * values[n - 1] - (n - 1) * values[n - 2]) / n);
  return values.slice(0, degree + 1);
}

function cholesky(matrix) {
  const n = matrix.length,
    l = Array.from({
      length: n
    }, () => Array(n).fill(0));
  for (let i = 0; i < n; i++)
    for (let j = 0; j <= i; j++) {
      let value = matrix[i][j];
      for (let k = 0; k < j; k++) value -= l[i][k] * l[j][k];
      l[i][j] = i === j ? Math.sqrt(value) : value / l[j][j];
    }
  return l;
}

function solveChol(l, b) {
  const n = b.length,
    y = Array(n).fill(0),
    x = Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    let v = b[i];
    for (let k = 0; k < i; k++) v -= l[i][k] * y[k];
    y[i] = v / l[i][i];
  }
  for (let i = n - 1; i >= 0; i--) {
    let v = y[i];
    for (let k = i + 1; k < n; k++) v -= l[k][i] * x[k];
    x[i] = v / l[i][i];
  }
  return x;
}

function bayesianPrediction(count, prior, noise) {
  const data = evidence.bayes,
    d = data.degree + 1,
    precision = Array.from({
      length: d
    }, (_, i) => Array.from({
      length: d
    }, (_, j) => i === j ? 1 / prior : 0)),
    rhs = Array(d).fill(0);
  for (let i = 0; i < count; i++) {
    const a = legendre(data.x[i], data.degree);
    for (let j = 0; j < d; j++) {
      rhs[j] += a[j] * data.y[i] / noise ** 2;
      for (let k = 0; k < d; k++) precision[j][k] += a[j] * a[k] / noise ** 2;
    }
  }
  const l = cholesky(precision),
    w = solveChol(l, rhs);
  return data.grid.map(x => {
    const a = legendre(x, data.degree),
      covA = solveChol(l, a),
      mu = a.reduce((v, b, i) => v + b * w[i], 0),
      variance = noise ** 2 + a.reduce((v, b, i) => v + b * covA[i], 0),
      sigma = Math.sqrt(Math.max(variance, 0));
    return {
      x,
      y: mu,
      lo: mu - 1.96 * sigma,
      hi: mu + 1.96 * sigma,
      sd: sigma
    };
  });
}

function bayesian() {
  heading("Bayesian updating and predictive uncertainty", "Change the data available to a Gaussian polynomial model and its stated assumptions. The posterior is solved exactly in the browser.");
  const controls = el("div", undefined, "controls");
  const [nl, count] = slider("Observed samples", 6, 40, 40);
  const [pl, prior] = slider("Prior coefficient variance", .05, 5, 1, .05);
  const [sl, noise] = slider("Assumed noise standard deviation", .05, .4, .15, .01);
  controls.append(nl, pl, sl);
  content.append(controls);
  const results = el("div");
  content.append(results);

  function render() {
    const predictions = bayesianPrediction(Number(count.value), Number(prior.value), Number(noise.value));
    const box = panel("Posterior prediction and95% Gaussian predictive bounds");
    const observed = evidence.bayes.x.slice(0, Number(count.value)).map((x, i) => ({
      x,
      y: evidence.bayes.y[i]
    }));
    box.append(plot([{
      label: "Posterior mean",
      points: predictions
    }, {
      label: "Lower predictive bound",
      points: predictions.map(p => ({
        x: p.x,
        y: p.lo
      }))
    }, {
      label: "Upper predictive bound",
      points: predictions.map(p => ({
        x: p.x,
        y: p.hi
      }))
    }, {
      label: "True function",
      points: predictions.map(p => ({
        x: p.x,
        y: Math.sin(3 * p.x)
      })),
      color: "#999"
    }], {
      xlabel: "Input",
      ylabel: "Output"
    }), plot([{
      label: "Observed data",
      points: observed
    }], {
      points: true,
      xlabel: "Input",
      ylabel: "Observed output"
    }), el("p", `${count.value} fixed-order observations · prior variance ${fmt(prior.value,2)} · noise variance ${fmt(Number(noise.value)**2,4)}.`, "muted"));
    results.replaceChildren(box);
  }
  [count, prior, noise].forEach(node => node.addEventListener("input", render));
  render();
  note("The degree-five Legendre model uses an isotropic Gaussian coefficient prior and known assumed observation noise. Predictive bounds include coefficient uncertainty and observation noise. Changing assumptions changes the posterior; these bounds are not a model-free guarantee under misspecification. Full-data default predictions are checked against the Python reference.");
}

function generalization() {
  heading("Generalization, complexity and kernels", "Inspect repeated population-risk experiments, exact and approximate kernel learning curves, and the implicit bias of logistic gradient descent.");
  const controls = el("div", undefined, "controls");
  const penalties = [...new Set(evidence.generalization.double_descent.map(r => r.penalty))];
  const [pl, penalty] = select("Ridge penalty", penalties);
  controls.append(pl);
  content.append(controls);
  const results = el("div");
  content.append(results);

  function render() {
    results.replaceChildren();
    const rows = evidence.generalization.double_descent.filter(r => r.penalty === Number(penalty.value));
    const ratios = [...new Set(rows.map(r => r.ratio))].sort((a, b) => a - b);
    const data = ratios.map(r => ({
      x: r,
      y: mean(rows.filter(f => f.ratio === r).map(f => f.population_mse))
    }));
    const box = panel("Model complexity versus population prediction risk");
    box.append(plot([{
      label: "Mean exact population MSE",
      points: data
    }], {
      xlabel: "Features / fitting observations",
      ylabel: "Population MSE",
      logY: true
    }), el("p", `${new Set(rows.map(r=>r.seed)).size} fitting repeats; noise variance 0.25. Risk is calculated from the known Gaussian population rather than another sampled test set.`, "muted"));
    results.append(box);
    const kernels = panel("Kernel methods with limited labels");
    const sizes = [...new Set(evidence.kernels.rows.map(r => r.train_size))].sort((a, b) => a - b);
    const series = [...new Set(evidence.kernels.rows.map(r => r.model))].map(model => ({
      label: name(model),
      points: sizes.map(size => ({
        x: size,
        y: mean(evidence.kernels.rows.filter(r => r.model === model && r.train_size === size).map(r => r.accuracy))
      }))
    }));
    kernels.append(plot(series, {
      xlabel: "Training labels",
      ylabel: "Held-out digit accuracy"
    }), legend(series));
    results.append(kernels);
    const implicit = panel("Implicit bias of logistic gradient descent");
    const history = evidence.generalization.implicit_bias;
    implicit.append(plot([{
      label: "Cosine to hard-margin direction",
      points: history.map(r => ({
        x: r.step,
        y: r.direction_cosine
      }))
    }], {
      xlabel: "Gradient step",
      ylabel: "Direction alignment"
    }), el("p", `Final weight norm ${fmt(history.at(-1).norm)}; loss ${fmt(history.at(-1).loss,5)}. A decreasing loss and increasing parameter norm can coexist with a converging direction.`, "muted"));
    results.append(implicit);
  }
  penalty.addEventListener("change", render);
  render();
  note("Regularization choices shown here are a controlled population study, not test-selected deployment settings. The kernel comparison has fixed bandwidth profiles and validation-selected regularization. Its digit accuracy does not establish competitiveness on modern large-image tasks.");
}

function representations() {
  heading("Representation learning and information budgets", "Compare random features, contrastive pretraining, scratch supervision, frozen ImageNet transfer and fine-tuned ConvNeXt. External pretraining and label budgets are explicitly different.");
  const rows = evidence.representations;
  const controls = el("div", undefined, "controls");
  const [ml, model] = select("Representation", [...new Set(rows.map(r => r.model))], "convnext_tiny_finetuned");
  const [sl, seed] = select("Fitting seed", [0, 1, 2]);
  controls.append(ml, sl);
  content.append(controls);
  const results = el("div");
  content.append(results);

  function render() {
    const available = rows.filter(r => r.model === model.value);
    const actualSeeds = available.map(r => r.seed);
    [...seed.options].forEach(o => {
      o.disabled = !actualSeeds.includes(Number(o.value));
    });
    if (!available.some(r => r.seed === Number(seed.value))) seed.value = String(available[0].seed);
    const row = available.find(r => r.seed === Number(seed.value));
    results.replaceChildren();
    const box = panel("Class-specific performance");
    box.append(table(["Class", "Accuracy"], ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"].map((label, i) => [label, `${fmt(100*row.per_class_accuracy[i],1)}%`])));
    const top = el("div", undefined, "metrics");
    [
      ["CIFAR-10 test accuracy", `${fmt(100*row.accuracy,2)}%`],
      ["Calibrated NLL", fmt(row.nll)],
      ["Fitting labels", row.labels.toLocaleString()],
      ["Prediction-set coverage", `${fmt(100*row.set_coverage,1)}%`]
    ].forEach(([a, b]) => {
      const card = el("div", undefined, "metric");
      card.append(el("strong", b), el("span", a));
      top.append(card);
    });
    results.append(top, box);
    const comparison = panel("Compare representation profiles");
    comparison.append(table(["Profile", "Fits", "Fitting labels", "Mean accuracy ± fit SD", "Mean calibrated NLL"], aggregate(rows, "accuracy").sort((a, b) => b.mean - a.mean).map(r => [name(r.method), r.fits.length, r.fits[0].labels.toLocaleString(), `${fmt(100*r.mean,2)} ± ${fmt(100*r.sd,2)}%`, fmt(mean(r.fits.map(f => f.nll)))])));
    if (row.gaussian_noise_accuracy !== undefined) comparison.append(el("p", `This fit under Gaussian pixel noise: ${fmt(100*row.gaussian_noise_accuracy,2)}% accuracy.`, "muted"));
    results.append(comparison);
  }
  [model, seed].forEach(node => node.addEventListener("change", render));
  render();
  note("Some profiles have one fitted seed; others have three. Contrastive pretraining uses additional unlabeled images; transfer uses external ImageNet initialization. ConvNeXt also has a separate selection/calibration label allocation. Noise failures are preserved; this is not an equal-information ranking.");
}

function restoration() {
  heading("Image restoration and conditional generation", "Inspect every recorded model, seed and stochastic draw for four fixed held-out photographs. Compare appearance with fidelity, perceptual metrics and observation consistency.");
  const rows = evidence.restoration;
  const controls = el("div", undefined, "controls");
  const [tl, task] = select("Task", [...new Set(rows.map(r => r.task))], "colorization");
  const [ml, model] = select("Model", []);
  const [sl, seed] = select("Fitting seed", []);
  const [dl, draw] = select("Generated draw", []);
  const [il, image] = select("Held-out image", [0, 1, 3, 5]);
  const [vl, comparison] = select("Compare prediction with", ["reference", "input"]);
  controls.append(tl, ml, sl, dl, il, vl);
  content.append(controls);
  const results = el("div");
  content.append(results);

  function setOptions(input, values) {
    const previous = input.value;
    input.replaceChildren();
    values.forEach(v => {
      const option = el("option", name(String(v)));
      option.value = String(v);
      input.append(option);
    });
    if (values.map(String).includes(previous)) input.value = previous;
  }

  function render() {
    const row = rows.find(r => r.task === task.value && r.method === model.value && r.seed === Number(seed.value) && r.draw === Number(draw.value) && r.image_index === Number(image.value));
    results.replaceChildren();
    const top = el("div", undefined, "metrics");
    [
      ["All-image mean PSNR", `${fmt(row.aggregate.psnr?.mean,2)} dB`],
      ["All-image mean LPIPS", fmt(row.aggregate.lpips?.mean)],
      ["Selected image PSNR", row.per_image.psnr === undefined ? "—" : `${fmt(row.per_image.psnr,2)} dB`],
      ["Selected image LPIPS", fmt(row.per_image.lpips)]
    ].forEach(([a, b]) => {
      const card = el("div", undefined, "metric");
      card.append(el("strong", b), el("span", a));
      top.append(card);
    });
    results.append(top);
    const box = panel("Fixed-image comparison");
    const grid = el("div", undefined, "images");
    [
      ["Observed input", row.input],
      ["Prediction", row.prediction],
      ["Reference photograph", row.truth]
    ].forEach(([label, path]) => {
      const figure = el("figure");
      const img = el("img");
      img.src = path;
      img.alt = label;
      figure.append(img, el("figcaption", label));
      grid.append(figure);
    });
    box.append(grid);
    const compare = el("div", undefined, "compare"),
      back = el("img"),
      front = el("img", undefined, "front"),
      divider = el("div", undefined, "divider");
    back.src = comparison.value === "reference" ? row.truth : row.input;
    back.alt = name(comparison.value);
    front.src = row.prediction;
    front.alt = "Prediction";
    compare.append(back, front, divider);
    const labels = el("div", undefined, "compare-labels");
    labels.append(el("span", "Prediction"), el("span", name(comparison.value)));
    const [ll, amount] = slider("Reveal prediction", 0, 100, 50);
    amount.addEventListener("input", () => {
      front.style.clipPath = `inset(0 ${100-Number(amount.value)}% 0 0)`;
      divider.style.left = `${amount.value}%`;
    });
    box.append(compare, labels, ll, el("p", row.profile, "muted"), link(row.metrics), document.createTextNode(" · "), link(row.source, "Original numerical gallery"));
    results.append(box);
    const unique = rows.filter(r => r.task === task.value && r.image_index === 0);
    const summary = panel("All methods on this task");
    summary.append(table(["Method", "Recorded seed/draw pairs", "Mean PSNR", "Mean LPIPS", "Mean SSIM"], [...new Set(unique.map(r => r.method))].map(method => {
      const selected = unique.filter(r => r.method === method);
      return [name(method), selected.length, fmt(mean(selected.map(r => r.aggregate.psnr.mean)), 2), fmt(mean(selected.map(r => r.aggregate.lpips.mean))), fmt(mean(selected.map(r => r.aggregate.ssim.mean)))];
    })));
    results.append(summary);
  }

  function draws() {
    const matching = rows.filter(r => r.task === task.value && r.method === model.value && r.seed === Number(seed.value));
    setOptions(draw, [...new Set(matching.map(r => r.draw))]);
    render();
  }

  function seeds() {
    setOptions(seed, [...new Set(rows.filter(r => r.task === task.value && r.method === model.value).map(r => r.seed))]);
    draws();
  }

  function methods() {
    setOptions(model, [...new Set(rows.filter(r => r.task === task.value).map(r => r.method))]);
    if (task.value === "colorization") model.value = "ddcolor";
    if (task.value === "super_resolution") model.value = "swinir";
    seeds();
  }
  task.addEventListener("change", methods);
  model.addEventListener("change", seeds);
  seed.addEventListener("change", draws);
  [draw, image, comparison].forEach(node => node.addEventListener("change", render));
  methods();
  note("Original models, external pretrained references and the controlled warm-start GAN study have different training histories. Compare their declared scopes. SR input previews display the bicubic enlargement of the actual 4× low-resolution observation. Photographs and draws are predetermined, not chosen for visual quality. Plausible generated color/detail can differ from the reference; diversity alone does not prove useful uncertainty.");
}

function signals() {
  heading("Signal equalization and nonlinear channel shift", "Compare an observed-symbol detector, linear and learned receivers, and a reference with known channel information.");
  const rows = evidence.signals.rows;
  const controls = el("div", undefined, "controls");
  const [gl, gamma] = select("Nonlinearity gamma", [...new Set(rows.map(r => r.gamma))]);
  controls.append(gl);
  content.append(controls);
  const results = el("div");
  content.append(results);

  function render() {
    const selected = rows.filter(r => r.gamma === Number(gamma.value));
    const snrs = [...new Set(selected.map(r => r.snr_db))].sort((a, b) => a - b);
    const series = [...new Set(selected.map(r => r.model))].map(model => ({
      label: name(model),
      points: snrs.map(snr => ({
        x: snr,
        y: mean(selected.filter(r => r.model === model && r.snr_db === snr).map(r => r.ber))
      }))
    }));
    const box = panel("Bit error rate across signal-to-noise ratios");
    box.append(plot(series, {
      xlabel: "SNR (dB)",
      ylabel: "Mean bit error rate"
    }), legend(series), table(["Receiver", "SNR", "Mean BER ± fit SD"], series.flatMap(s => s.points.map(p => {
      const fits = selected.filter(r => name(r.model) === s.label && r.snr_db === p.x);
      return [s.label, p.x, `${fmt(p.y,6)} ± ${fmt(sd(fits.map(r=>r.ber)),6)}`];
    }))));
    results.replaceChildren(box);
  }
  gamma.addEventListener("change", render);
  render();
  note("The known-channel reference has additional model information. Comparisons preserve fixed test blocks and three fitting seeds. Channel/noise shifts are sensitivity studies, not universal robustness guarantees.");
}

const sections = {
  Classification: classification,
  Regression: regression,
  Clustering: clustering,
  "Bayesian inference": bayesian,
  Generalization: generalization,
  Representations: representations,
  "Image restoration": restoration,
  "Signal equalization": signals
};
const nav = document.getElementById("concepts");

function show(title) {
  nav.querySelectorAll("button").forEach(button => button.setAttribute("aria-selected", String(button.textContent === title)));
  sections[title]();
  content.dataset.section = title;
}
Object.keys(sections).forEach(title => {
  const button = el("button", title);
  button.type = "button";
  button.setAttribute("role", "tab");
  button.setAttribute("aria-selected", "false");
  button.addEventListener("click", () => show(title));
  nav.append(button);
});
show("Image restoration");
