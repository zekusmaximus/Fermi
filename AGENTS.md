# AGENTS.md — Working Rules for This Repository

This repository implements an **empirical constraint engine and Bayesian Optimal Experimental Design (BOED) desk for the Fermi paradox**. Any coding agent, research agent, or human contributor working in this repo must follow these rules strictly.

---

## 1. Source of Truth Hierarchy
Before editing code, read in this order:
1. `SPEC.md` — Architectural invariants, schemas, and multi-milestone roadmap.
2. The active `docs/MILESTONE_*.md` — Exact scope and acceptance criteria for the current sprint.
3. Existing tests in `tests/test_invariants.py` — Executable proof of scientific invariants.

Do not invent scope outside the active milestone brief. If a milestone brief conflicts with `SPEC.md`, stop and flag the conflict rather than silently resolving it.

---

## 2. Non-Negotiable Scientific & Engineering Laws

1. **Never Fake Provenance or Data Tiers.**
   Every dataset, target table, or calibration curve MUST declare its `data_tier` (`published_catalog`, `reproduced_archive`, `illustrative_calibration`, or `synthetic_proxy`). Never label generated/synthetic star coordinates or placeholder sensitivities as published catalog data.
2. **Never Hardcode Outputs to Pass Tests.**
   Models must compute quantities from first-principles physics and input parameters. Never inject special-case branches for specific benchmark hypotheses (`frequent_slow_expanders`, `HIP 1692`, etc.).
3. **Preserve Dimensional Safety (`astropy.units`).**
   All physical quantities (distances, velocities, fluxes, luminosities, rates, timescales) must carry explicit `astropy.units` at module boundaries. Strip `.value` only inside vectorized NumPy/SciPy inner loops after explicit `.to(...)` conversion to documented SI/astro units.
4. **Never Bypass the Six Core Invariants.**
   Never weaken or skip tests for:
   - Zero-emission invariance (`p_detect == 0 => LR == 1.0`).
   - Non-double-counting (`pointing_group_id` uniqueness).
   - Tier-3 quarantine (unfalsifiable/monoculture hypotheses cannot receive empirical exclusion scores).
   - 2D vs. 3D geometry explicitness.
   - Deterministic seeded replay.
5. **No P-Hacking or "Surprise" Quotas.**
   Null results, weak separation, and failed screening rules are first-class scientific outputs. Never tune priors, thresholds, or seeds to manufacture a dramatic separation or "exciting" finding.
6. **Keep Macro and Micro Layers Decoupled.**
   Instrument-level calibration modules (e.g., AllWISE blend recovery, per-target radio beam profiles) must output clean likelihood / completeness / false-positive objects consumed by `inference/engine.py`. Never hard-code instrument calibration quirks inside the macro expansion kernel.

---

## 3. Development Environment & Commands

This project uses `uv` for deterministic Python environment management.

```bash
# Install / sync dependencies
uv sync

# Run full test suite (required before every handoff)
uv run pytest -v

# Run linter & formatter checks
uv run ruff check .
uv run ruff format --check .

# Validate schemas, units, and catalog checksums
uv run fermi-desk validate

# Run benchmark comparison
uv run fermi-desk compare --geometry thin_disk_2d --radio-mode flat_probability --seed 42
```

If adding a dependency, update `pyproject.toml` via `uv add` and justify why standard `numpy`/`scipy`/`astropy` cannot handle it.

---

## 4. Coding & Testing Discipline

- **Deterministic RNG only:** Never call bare `np.random.rand()` or `random.random()`. Always accept an explicit `rng: np.random.Generator` or `seed: int` and record the seed in the run receipt.
- **Explicit error types:** Raise domain-specific exceptions (`DoubleCountingError`, `Tier3ScoringError`, `UnitValidationError`, `ProvenanceError`) instead of generic `ValueError` where invariants are violated.
- **Small, composable functions:** Keep physics kernels pure and side-effect-free (no hidden file reads, no hidden network calls inside `models/`).
- **Network isolation in tests:** `uv run pytest` MUST run 100% offline using local fixtures in `tests/` and `data/`. Live archive queries (IRSA, Gaia, Berkeley SETI) belong in explicit scripts or CLI commands, never inside unit tests.

---

## 5. Required Handoff Format (For Planning Agent Audit)

At the end of every coding session or milestone, report back using this exact structure so results can be audited cleanly:

1. **Summary of Changes:** Bullet list of files created/modified and what each implements.
2. **Verification Output:** Verbatim stdout/stderr from `uv run pytest -v`, `uv run ruff check .`, and `uv run fermi-desk validate`.
3. **Benchmark Comparison Output:** Verbatim CLI output and key numbers from the primary `fermi-desk compare` runs (including receipt JSON path and `rng_seed`).
4. **Assumptions, Approximations & Placeholders:** Explicit list of any approximations made (e.g., analytic vs. Monte Carlo integration tolerances, synthetic vs. published catalogs, unverified constants).
5. **Open Questions / Next Risks:** Any scientific ambiguities or architectural friction encountered during implementation.