import { describe, expect, it } from "vitest";
import type { Gate2, Gate2Screen } from "../api";
import { SOURCE_LABEL, SOURCE_TIER, armARecoveryByFlux, defaultSource, matchedControl } from "./gate2";

const screen = (armA: number[][], armB: number[]): Gate2Screen => ({
  n_stars: 4,
  null_false_alarm: { k: 0, n: 4, rate: 0, clopper_pearson_95: [0, 0.6] },
  arm_a_blend_recovery: armA,
  arm_a_k: armA.map((r) => r.map((v) => v * 4)),
  arm_a_clopper_pearson_95: armA.map((r) => r.map(() => [0, 1] as [number, number])),
  arm_b_false_rejection: armB,
  arm_b_k: armB.map((v) => v * 4),
  arm_b_clopper_pearson_95: armB.map(() => [0, 1] as [number, number]),
  arm_b_false_rejection_pooled: { k: 0, n: 8, rate: 0, clopper_pearson_95: [0, 0.4] },
  blend_survival_within_3p25_arcsec: 0,
  centered_excess_retained_pooled: 1,
});

const gate = {
  separations_arcsec: [1, 3, 6],
  flux_fractions: [0.5, 0.9],
  screens: {
    centroid: screen([[0, 0.5], [0.5, 1], [1, 1]], [0, 0.25]),
    image: screen([[0, 0], [1, 1], [1, 1]], [0, 0]),
    naive_flux_excess: screen([[1, 1], [1, 1], [1, 1]], [1, 1]),
  },
} as unknown as Gate2;

describe("Gate #2 heatmap source toggle", () => {
  it("defaults to the real withheld grid only when it is pinned", () => {
    expect(defaultSource(gate)).toBe("real");
    expect(defaultSource(null)).toBe("synthetic");
    expect(SOURCE_LABEL.real).toBe("AllWISE W3 Atlas (Gate #2)");
    expect(SOURCE_LABEL.synthetic).toBe("Synthetic Gaussian PSF");
    expect(SOURCE_LABEL.unwise).toBe("unWISE W3 (Unsmoothed — Fresh Split)");
    // Tiers are never blended: both archive sources are cyan, the synthetic grid amber.
    expect(SOURCE_TIER).toEqual({
      real: "reproduced_archive",
      synthetic: "illustrative_calibration",
      unwise: "reproduced_archive",
    });
  });
});

describe("matched centered-excess control", () => {
  it("area-weights Arm A over theta <= 3.25 arcsec (p ~ theta)", () => {
    // theta 1 and 3 enter with weights 1/4 and 3/4; theta 6 is excluded.
    expect(armARecoveryByFlux(gate.screens.image.arm_a_blend_recovery, gate.separations_arcsec)).toEqual([0.75, 0.75]);
    expect(armARecoveryByFlux([], [1])).toEqual([]);
  });
  it("gives the naive bright-excess rule zero net discrimination", () => {
    const naive = matchedControl(gate, "naive_flux_excess");
    expect(naive.map((c) => c.armA)).toEqual([1, 1]);
    expect(naive.map((c) => c.armB)).toEqual([1, 1]);
    expect(naive.every((c) => c.net === 0)).toBe(true);
    const img = matchedControl(gate, "image");
    expect(img.map((c) => c.net)).toEqual([0.75, 0.75]);
    expect(matchedControl(gate, "centroid")[1]).toMatchObject({ fBg: 0.9, armB: 0.25, armBk: 1, n: 4 });
  });
});
