"""Exercise all concept sections and measured presets in a real local browser."""

import argparse
from pathlib import Path

import numpy as np
from playwright.sync_api import sync_playwright

from learning_lab.artifacts import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-pending", action="store_true")
    args = parser.parse_args()
    errors = []
    presets = 0
    path = Path("docs/explorer.html")
    preview = Path("assets/previews")
    preview.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True, args=["--no-sandbox"])
        page = browser.new_page(viewport=dict(width=1280, height=960))
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(path.resolve().as_uri())
        page.wait_for_function(
            'document.querySelectorAll("#content img").length>0 && [...document.querySelectorAll("#content img")].every(i=>i.complete&&i.naturalWidth>0)'
        )
        completion = page.evaluate("evidence.completion.status")
        assert args.allow_pending or completion == "complete"
        assert page.locator("#concepts button").count() == 8
        for section in page.locator("#concepts button").all_text_contents():
            page.get_by_role("tab", name=section, exact=True).click()
            assert page.locator("#content").get_attribute("data-section") == section
            assert page.locator("#content h2").count() >= 2
            controls = page.locator("#content select")
            if section in ["Classification", "Regression", "Clustering", "Representations"]:
                if section == "Classification":
                    threshold = page.locator('input[data-control="Exploratory test threshold"]')
                    assert threshold.input_value() == "0.5"
                    assert page.get_by_text("Threshold 0.50", exact=False).count() == 1
                    threshold.evaluate(
                        "n=>{n.value='0.8';n.dispatchEvent(new Event('input',{bubbles:true}));}"
                    )
                    assert threshold.locator("..").locator("span").inner_text() == "0.8"
                    assert page.get_by_text("Threshold 0.80", exact=False).count() == 1
                    threshold.evaluate(
                        "n=>{n.value='0.5';n.dispatchEvent(new Event('input',{bubbles:true}));}"
                    )
                if section in ["Classification", "Regression", "Clustering"]:
                    cases = controls.nth(0).locator("option").evaluate_all("xs=>xs.map(x=>x.value)")
                    for case in cases:
                        controls.nth(0).select_option(case)
                        methods = (
                            controls.nth(1).locator("option").evaluate_all("xs=>xs.map(x=>x.value)")
                        )
                        for method in methods:
                            controls.nth(1).select_option(method)
                            assert page.locator("#content svg").count() >= 1
                            presets += 1
                else:
                    for method in (
                        controls.nth(0).locator("option").evaluate_all("xs=>xs.map(x=>x.value)")
                    ):
                        controls.nth(0).select_option(method)
                        assert page.locator("#content table").count() == 2
                        presets += 1
            elif section == "Image restoration":
                for task in (
                    controls.nth(0).locator("option").evaluate_all("xs=>xs.map(x=>x.value)")
                ):
                    controls.nth(0).select_option(task)
                    for method in (
                        controls.nth(1).locator("option").evaluate_all("xs=>xs.map(x=>x.value)")
                    ):
                        controls.nth(1).select_option(method)
                        for seed in (
                            controls.nth(2).locator("option").evaluate_all("xs=>xs.map(x=>x.value)")
                        ):
                            controls.nth(2).select_option(seed)
                            for draw in (
                                controls.nth(3)
                                .locator("option")
                                .evaluate_all("xs=>xs.map(x=>x.value)")
                            ):
                                controls.nth(3).select_option(draw)
                                page.wait_for_function(
                                    '[...document.querySelectorAll("#content img")].every(i=>i.complete&&i.naturalWidth>0)'
                                )
                                assert page.locator("#content img").count() == 5
                                presets += 1
                page.locator('input[data-control="Reveal prediction"]').evaluate(
                    "(node)=>{node.value='80';node.dispatchEvent(new Event('input',{bubbles:true}));}"
                )
                assert (
                    page.locator("img.front").evaluate("x=>x.style.clipPath")
                    == "inset(0px 20% 0px 0px)"
                )
            elif section == "Bayesian inference":
                assert page.locator('input[data-control="Observed samples"]').input_value() == "40"
                assert (
                    page.locator('input[data-control="Prior coefficient variance"]').input_value()
                    == "1"
                )
                assert (
                    page.locator(
                        'input[data-control="Assumed noise standard deviation"]'
                    ).input_value()
                    == "0.15"
                )
                values = page.evaluate("bayesianPrediction(40,1,.15)")
                reference = page.evaluate("evidence.bayes")
                np.testing.assert_allclose(
                    [r["y"] for r in values], reference["reference_mean"], rtol=0, atol=1e-9
                )
                np.testing.assert_allclose(
                    [r["sd"] for r in values], reference["reference_sd"], rtol=0, atol=1e-9
                )
                page.locator('input[data-control="Observed samples"]').evaluate(
                    "(node)=>{node.value='12';node.dispatchEvent(new Event('input',{bubbles:true}));}"
                )
                assert page.locator("#content svg").count() == 2
                assert all(
                    np.isfinite(list(r.values())).all()
                    for r in page.evaluate("bayesianPrediction(12,.05,.4)")
                )
            page.set_viewport_size(dict(width=390, height=844))
            assert page.evaluate("document.documentElement.scrollWidth<=innerWidth"), section
            page.set_viewport_size(dict(width=1280, height=960))
        page.get_by_role("tab", name="Classification", exact=True).click()
        page.screenshot(path=str(preview / "classification_desktop.png"), full_page=True)
        page.get_by_role("tab", name="Image restoration", exact=True).click()
        page.wait_for_function(
            '[...document.querySelectorAll("#content img")].every(i=>i.complete&&i.naturalWidth>0)'
        )
        page.screenshot(path=str(preview / "restoration_desktop.png"), full_page=True)
        page.set_viewport_size(dict(width=390, height=844))
        page.screenshot(path=str(preview / "restoration_mobile.png"), full_page=True)
        browser.close()
    assert not errors, errors
    write_json(
        "docs/explorer_browser_validation.json",
        dict(
            status="development" if completion != "complete" else "passed",
            sections=8,
            checked_presets=presets,
            bayesian_python_agreement_absolute_tolerance=1e-9,
            mobile_width=390,
            javascript_errors=errors,
        ),
    )
    print(
        "Browser passed:",
        presets,
        "presets;8 concepts; analytic Bayesian agreement; mobile layouts",
    )


if __name__ == "__main__":
    main()
