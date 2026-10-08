# User interaction audit

The explorer was exercised in Chromium over a local HTTP server, using the same
file layout as a normal browser preview. The audit selects every enabled finite
combination of dropdown values, including dependent model, fitting-seed and image
choices. It checks slider endpoints and middle values; short discrete sliders
with at most thirteen positions are checked at every position. Multiple sliders
are checked jointly on that grid. This is a finite validation grid, rather than a
claim to test every possible continuous parameter or browser implementation.

At each selection, the checker verifies that visible images load, their HTTP
bytes match the repository assets, local evidence links exist, tables have data,
and JavaScript produces no errors or invalid numeric text. The local numerical
oracle checks selected fields and asset associations against the embedded
scientific evidence. The lab-specific validators independently bind that evidence
to source records and numerical arrays. A 390-pixel viewport is checked in each
concept area.

From the repository root, with the usual package/development dependencies installed:

```bash
python -m pip install playwright==1.63.0
python -m playwright install chromium
python scripts/check_demo_controls.py
python scripts/check_explorer_browser.py
```

The HTTP checker starts and stops its own temporary localhost server and does not
use the ports of existing preview servers. Playwright and Chromium are development
verification tools; viewers only need a browser. `PLAYWRIGHT_BROWSERS_PATH` can
point to a personal external browser cache.

[HTTP interaction results](demo_controls_validation.json) list the exact tested
selector combinations, slider-check count, image files and local links.
[Scientific browser results](explorer_browser_validation.json) record the separate
numerical oracle checks. The reports identify the tested HTML by its SHA-256 digest.
After changing HTML or assets, rerun the relevant checks before publishing.

The checks include every held-out restoration photograph and both overlay
comparison modes, every available fitting seed, and stochastic draw. The
Bayesian controls are checked against an independent NumPy posterior solve.
`python scripts/validate_explorer.py` also checks every embedded prediction row
and every public restoration PNG against its original numerical gallery.
