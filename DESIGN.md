# UI Legibility, Visualization & Experience Bible (`DESIGN.md`)

Living design contract for `web/` (the Fermi Paradox Empirical Constraint Desk frontend). Read alongside `AGENTS.md`, `SPEC.md`, and `docs/PLANNING_LEDGER.md` before modifying any frontend view, copy, or visual component.

---

## 1. Design North Star & Non-Negotiable Visual Laws

`fermi-desk` is a **scientific instrument and live observatory**, not a marketing landing page or a Drake Equation toy. It should feel immediately legible to a curious newcomer while remaining unimpeachable to a working astronomer.

1. **Instrument Aesthetic:** High-contrast dark/slate observatory palette, crisp borders, and monospace `tabular-nums` on every probability, `log10 BF`, coordinate, distance, and SHA-256 hash.
2. **Strict Data-Tier Color Separation (Never Blend):**
   - `published_catalog` → **Emerald**
   - `reproduced_archive` → **Cyan**
   - `illustrative_calibration` → **Amber**
   - `quarantined` (Tier 3) → **Muted Slate / Violet-Gray** with explicit `ZERO EMPIRICAL SCORE` tag
   - Active warnings (`LOCAL APPROX VIOLATED`, `ILLUSTRATIVE CHANNELS INCLUDED`) → **High-Contrast Red/Amber**
3. **100% Offline & Lightweight Native Visuals:**
   - All charts, schematics, and micro-illustrations use **native SVG/HTML/CSS** (zero external CDN fonts/images, zero heavy charting bundles, zero outbound network requests).
   - Never use `dangerouslySetInnerHTML`.
4. **Progressive Disclosure Over Walls of Text:**
   - Keep `Overview` tight as Mission Control.
   - Explain controls *at the point of interaction* rather than forcing users to hunt through a separate glossary page.

---

## 2. Control & Toggle Legibility Standard

Every interactive control added to `Macro Desk`, `Radio`, `Infrared`, or `Action Queue` MUST ship with:

1. **Plain-English Label:** Avoid raw snake_case enum strings as the sole visible label (keep the exact API/CLI enum visible in subtle monospace beneath or inside the tooltip for reproducibility).
2. **1-Line Dynamic Caption:** A concise sentence directly under the control group that updates immediately when the active option changes, stating *what physically changes* and *whether it affects the verified headline*.
3. **Inline `?` Popover:** A compact trigger explaining the physical trade-off, key citation/invariant, and CLI flag equivalent (`--geometry`, `--radio-mode`, `--pointing-merge-policy`, etc.).

### Global "How to Read This Desk" Reference Drawer (Planned)
A single slide-over drawer accessible from the persistent header covering four recurring concepts in plain English:
- **Bayes Factors (`log10 BF`) & Jeffreys' Scale:** Why `|log10 BF| < 0.5` means verified data cannot separate the two regimes (and why null verdicts are first-class results).
- **Data Tiers:** Exact meaning of emerald (`published_catalog`), cyan (`reproduced_archive`), and amber (`illustrative_calibration`) badges.
- **Earth-Unreached Conditioning:** Why conditioning on Earth still being untouched changes how many settled neighbors you expect nearby.
- **Tier-3 Monoculture Quarantine:** Why unfalsifiable sociological hypotheses (*Zoo*, *Dark Forest*) carry zero empirical score and require a compliance-fragility audit.

---

## 3. Visualization Registry & Build Roadmap

Visualizations replace dense numeric tables with immediate physical intuition using data the backend already returns.

| ID | View | Visual | Status | Target Sprint |
|---|---|---|---|---|
| `V1` | Radio Drill-Down | 692-Target Distance vs. `EIRP_min` Log-Log Scatter (with HIP 1692 highlight) | **Shipped** (`v0.1.0`) | Done |
| `V2` | Infrared Drill-Down | Angular Separation × Background Flux Blend-Recovery Heatmap Matrix | **Shipped** (`v0.1.0` / `v0.1.2`) | Done |
| `V3` | Macro Desk | Inline `?` Popovers + 1-Line Dynamic Captions for all controls (`geometry`, `radio_mode`, `disk_profile`, `T` marginalization, `pointing_merge_policy`, `earth_unreached`) | **Planned (Incremental)** | Start `v0.2.1b`, maintain every sprint |
| `V4` | Macro Desk | Diverging `log10 BF` Channel Bar on Jeffreys Scale (`±0.5`, `±1`, `±2`), visually separating verified vs. illustrative channels | **Planned** | Pre-launch UI polish (`v0.2.3`) |
| `V5` | Macro Desk | Reactive Galactic Disk & Bubble Schematic (top-down + edge-on SVG showing Earth at `8 kpc`, Frequent/Slow vs. Rare/Fast fronts, and the `T_valid = 52.2 kyr` cliff) | **Planned** | Pre-launch UI polish (`v0.2.3`) |
| `V6` | Macro Desk / Queue | Lookback-Time (`T`) Sweep Curve with `T_valid = 52.2 kyr` vertical cutoff | **Planned** | Pre-launch UI polish (`v0.2.3`) |
| `V7` | Radio Drill-Down | GBT `8.4'` Off-Axis Beam Roll-Off $B(\theta)$ + $d^2$ Attenuation Visual (explaining why 288k background stars collapse from `0.827` to `0.0145` bits) | **Planned** | `v0.2.1b` or `v0.2.3` |
| `V8` | Infrared Drill-Down | 1D PSF Cross-Section (`1"` galaxy inside `13.69"` Atlas PSF vs. `unWISE`) + 15-Point Contamination Flip Strip (Suazo vs. Zackrisson crossing `LR = 1.0`) | **Planned** | Post-`unWISE` (`v0.2.2` / `v0.2.3`) |
| `V9` | Action Queue | BOED Priority Scatter (`Expected EIG Bits` vs. `Effort Hours` with diagonal `bits/hr` contours) | **Planned** | Pre-launch UI polish (`v0.2.3`) |

---

## 4. Easter Egg Policy & Registry

Playful details are welcome so curious visitors and BYOA contributors enjoy exploring the desk, subject to four strict guardrails:

1. **Zero Scientific Interference:** Easter eggs must never alter model outputs, API payloads, SHA-256 receipts, or `data_tier` badges.
2. **Opt-In / Discovered Only:** Never block first-load reading or cover tables/controls unprompted.
3. **Accessible & Motion-Safe:** Honor `@media (prefers-reduced-motion: reduce)` and allow instant dismissal via click or `Escape`.
4. **Offline Native Assets Only:** Tiny inline SVG + CSS + console text only; zero third-party images or network calls.

### Planned Easter Eggs
- **`E1` — "I'm Not Here" Zoo Alien:** Triple-clicking the Tier-3 `QUARANTINED — ZERO EMPIRICAL SCORE` badge (or typing `zoo` outside text inputs) peeks a tiny monochrome SVG alien over the Zoo Hypothesis card saying *"I'm not here. (Compliance: 100.0%. Please do not notify the Adversary.)"*; clicking it retreats with `log10 BF = 0.000000 (still quarantined)`.
- **`E2` — DevTools 1.42 GHz Console Dish:** One-time boot greeting in the browser console with an ASCII radio dish pointing developers and agent builders to `AGENTS.md` and `GET /api/queue`.
- **`E3` — Los Alamos 1950 Pulse:** Double-clicking the header observatory glyph runs a single subtle radar sweep and displays *"Where is everybody? — Los Alamos, 1950 (still 0 detections)."*
- **`E4` — HIP 1692 Neighborhood Wink:** Clicking the `1,176×` badge reveals a subtle footnote: *"23 pc and 789 pc are slightly different neighborhoods."*

---

## 5. UI Change Verification Checklist

Every PR touching `web/` must verify:
- [ ] `npm run typecheck`, `npm test`, and `npm run build` succeed cleanly.
- [ ] Zero console errors, zero failed requests, and zero outbound external requests in headless verification.
- [ ] Every new toggle has a plain-English label, dynamic 1-line caption, and `?` explanation.
- [ ] All `data_tier` colors match Section 1 and illustrative data never visually merges into verified bounds.
- [ ] Any Easter egg passes `prefers-reduced-motion`, leaves receipts/numbers untouched, and is dismissible.