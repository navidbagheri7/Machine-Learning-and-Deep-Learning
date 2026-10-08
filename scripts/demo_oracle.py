"""Source-bound numerical assertions for the currently selected demo controls."""

import math
import re

import numpy as np

_CACHE = None


def close(text, expected):
    if expected is None:
        assert text == "—", (text, expected)
        return
    if isinstance(expected, str):
        assert text == expected, (text, expected)
        return
    found = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?(?:e[-+]?\d+)?", text, re.I)
    assert found, (text, expected)
    token = found.group().replace(",", "")
    places = len(token.split("e")[0].partition(".")[2])
    exponent = int(token.lower().partition("e")[2] or "0")
    tolerance = 0.501 * 10 ** (exponent - places)
    assert math.isclose(float(token), expected, rel_tol=1e-12, abs_tol=tolerance), (text, expected)


def cards(page, selector, expected):
    actual = page.locator(selector).all_text_contents()
    assert len(actual) == len(expected), (actual, expected)
    for text, value in zip(actual, expected, strict=True):
        close(text, value)


def image_sources(page, selector, expected):
    actual = page.locator(selector).evaluate_all('xs=>xs.map(x=>x.getAttribute("src"))')
    assert actual == expected, (actual, expected)


def check(page):
    global _CACHE
    if _CACHE is None:
        _CACHE = page.evaluate("evidence")
    e = _CACHE
    section = page.locator("#content").get_attribute("data-section")
    c = page.locator("#content select").evaluate_all(
        "xs=>Object.fromEntries(xs.map(x=>[x.dataset.control,x.value]))"
    )
    selector = "#content .metric strong"
    expected = None
    if section == "Classification":
        r = next(
            r
            for r in e["classification"]
            if r["dataset"] == c["Dataset"]
            and r["model"] == c["Model"]
            and r["seed"] == int(c["Fitting seed"])
        )
        expected = [100 * r["accuracy"], r["log_loss"], 100 * r["set_coverage"], r["n"]]
        if r["thresholds"]:
            value = float(
                page.locator('input[data-control="Exploratory test threshold"]').input_value()
            )
            t = r["thresholds"][round(value * 100)]
            cells = page.locator("#content table").first.locator("tr:has(td)").all()
            for element, reference in zip(cells, [[t["tn"], t["fp"]], [t["fn"], t["tp"]]]):
                for text, number in zip(element.locator("td").all_text_contents()[1:], reference):
                    close(text, number)
    elif section == "Regression":
        r = next(
            r
            for r in e["regression"]
            if r["protocol"] == c["Test protocol"]
            and r["model"] == c["Model"]
            and r["seed"] == int(c["Fitting seed"])
        )
        expected = [r["rmse"], r["mae"], 100 * r["interval_coverage"], r["interval_width"]]
    elif section == "Clustering":
        r = next(
            r
            for r in e["clustering"]["rows"]
            if r["case"] == c["Dataset"]
            and r["method"] == c["Method"]
            and r["seed"] == int(c["Fitting seed"])
        )
        expected = [
            r["ari_all_points"],
            100 * r["nonnoise_fraction"],
            r["clusters"],
            r["silhouette_nonnoise"],
        ]
    elif section == "Representations":
        r = next(
            r
            for r in e["representations"]
            if r["model"] == c["Representation"] and r["seed"] == int(c["Fitting seed"])
        )
        expected = [100 * r["accuracy"], r["nll"], r["labels"], 100 * r["set_coverage"]]
    elif section == "Image restoration":
        r = next(
            r
            for r in e["restoration"]
            if r["task"] == c["Task"]
            and r["method"] == c["Model"]
            and r["seed"] == int(c["Fitting seed"])
            and r["draw"] == int(c["Generated draw"])
            and r["image_index"] == int(c["Held-out image"])
        )
        expected = [
            r["aggregate"]["psnr"]["mean"],
            r["aggregate"]["lpips"]["mean"],
            r["per_image"].get("psnr"),
            r["per_image"].get("lpips"),
        ]
        image_sources(
            page,
            "#content img",
            [
                r["input"],
                r["prediction"],
                r["truth"],
                r["truth"] if c["Compare prediction with"] == "reference" else r["input"],
                r["prediction"],
            ],
        )
        value = float(page.locator('input[data-control="Reveal prediction"]').input_value())
        clip = page.locator("img.front").evaluate("x=>getComputedStyle(x).clipPath")
        assert clip in [f"inset(0px {100 - value:g}% 0px 0px)", f"inset(0px {100 - value:g}%)"], (
            clip
        )
    elif section == "Bayesian inference":
        values = [
            float(page.locator(f'input[data-control="{key}"]').input_value())
            for key in [
                "Observed samples",
                "Prior coefficient variance",
                "Assumed noise standard deviation",
            ]
        ]
        n, prior, noise = values
        n = int(n)
        b = e["bayes"]
        phi = np.polynomial.legendre.legvander(b["x"][:n], b["degree"])
        test = np.polynomial.legendre.legvander(b["grid"], b["degree"])
        precision = phi.T @ phi / noise**2 + np.eye(b["degree"] + 1) / prior
        weights = np.linalg.solve(precision, phi.T @ np.asarray(b["y"][:n]) / noise**2)
        means = test @ weights
        sd = np.sqrt(noise**2 + np.einsum("ij,ji->i", test, np.linalg.solve(precision, test.T)))
        actual = page.evaluate("([n,p,s])=>bayesianPrediction(n,p,s)", values)
        np.testing.assert_allclose([r["y"] for r in actual], means, rtol=0, atol=1e-9)
        np.testing.assert_allclose([r["sd"] for r in actual], sd, rtol=0, atol=1e-9)
    if expected is not None:
        cards(page, selector, expected)
