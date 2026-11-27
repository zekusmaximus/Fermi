import { describe, expect, it } from "vitest";
import {
  DISK_PROFILE_LABELS,
  DISK_PROFILE_OPTIONS,
  GEOMETRY_OPTIONS,
  MARGINALIZE_LABEL,
  geometryRequest,
  isFiniteGeometry,
  localApproxBadge,
  lookbackSummary,
  showDiskProfile,
} from "./geometry";

describe("geometry selector options", () => {
  it("offers finite_disk_3d as the third geometry beside the two local disks", () => {
    expect(GEOMETRY_OPTIONS).toEqual(["thin_disk_2d", "thick_disk_3d", "finite_disk_3d"]);
  });
  it("labels the two disk profiles", () => {
    expect(DISK_PROFILE_OPTIONS).toEqual(["uniform", "exponential"]);
    expect(DISK_PROFILE_LABELS.uniform).toBe("Uniform Slab");
    expect(DISK_PROFILE_LABELS.exponential).toBe("Exponential Disk");
  });
  it("reveals the disk-profile toggle only for the finite disk", () => {
    expect(showDiskProfile("finite_disk_3d")).toBe(true);
    expect(showDiskProfile("thin_disk_2d")).toBe(false);
    expect(showDiskProfile("thick_disk_3d")).toBe(false);
    expect(showDiskProfile(undefined)).toBe(false);
    expect(isFiniteGeometry("finite_disk_3d")).toBe(true);
    expect(isFiniteGeometry("disk")).toBe(false);
  });
});

describe("local-approximation validity badge", () => {
  it("is red LOCAL APPROX VIOLATED with T_valid under an infinite disk", () => {
    const b = localApproxBadge({ local_approximation_valid: false, t_valid_kyr: 52.185 });
    expect(b).toMatchObject({ kind: "violated", label: "LOCAL APPROX VIOLATED", className: "badge-violated" });
    expect(b?.tValidKyr).toBeCloseTo(52.185);
  });
  it("clears to VALID (FINITE GALAXY) when the receipt reports the finite disk", () => {
    const b = localApproxBadge({ local_approximation_valid: true, local_approx: "valid_finite_galaxy", t_valid_kyr: null });
    expect(b).toMatchObject({ kind: "finite", label: "VALID (FINITE GALAXY)", className: "badge-finite", tValidKyr: null });
  });
  it("is a plain ok inside the infinite-disk validity radius", () => {
    expect(localApproxBadge({ local_approximation_valid: true, t_valid_kyr: 5218.5 })).toMatchObject({ kind: "ok", label: "ok" });
  });
  it("never shows the finite badge unless the server says so", () => {
    // a violated infinite-disk result stays violated whatever geometry the user has selected
    expect(localApproxBadge({ local_approximation_valid: false })?.kind).toBe("violated");
    expect(localApproxBadge({ local_approximation_valid: true, local_approx: "VIOLATED" })?.kind).toBe("violated");
    expect(localApproxBadge({ local_approximation_valid: false, local_approx: "something_else" })?.kind).toBe("violated");
  });
  it("drops a non-finite T_valid and handles a hypothesis without expansion", () => {
    expect(localApproxBadge({ local_approximation_valid: false, t_valid_kyr: Number.NaN })?.tValidKyr).toBeNull();
    expect(localApproxBadge(null)).toBeNull();
    expect(localApproxBadge(undefined)).toBeNull();
  });
});

describe("compare request geometry fields", () => {
  it("keeps the landing defaults identical to v0.1.2", () => {
    expect(geometryRequest({ geometry: "thin_disk_2d", diskProfile: "uniform", marginalizeLookback: false })).toEqual({
      geometry_mode: "thin_disk_2d",
      disk_profile: "uniform",
      marginalize_lookback: false,
    });
  });
  it("sends the selected profile only for the finite disk", () => {
    expect(geometryRequest({ geometry: "finite_disk_3d", diskProfile: "exponential", marginalizeLookback: false }).disk_profile).toBe(
      "exponential",
    );
    expect(geometryRequest({ geometry: "thick_disk_3d", diskProfile: "exponential", marginalizeLookback: true })).toEqual({
      geometry_mode: "thick_disk_3d",
      disk_profile: "uniform",
      marginalize_lookback: true,
    });
  });
});

describe("lookback marginalisation labels", () => {
  it("uses the exact toggle wording", () => {
    expect(MARGINALIZE_LABEL).toBe("Marginalize Lookback T (50 kyr–10 Myr)");
  });
  it("summarises the prior from the receipt, or says T is fixed", () => {
    expect(lookbackSummary(null)).toBe("lookback T fixed at the registry default");
    expect(
      lookbackSummary({
        prior: { effective_bounds_gyr: [5e-5, 1e-2] },
        quadrature: { n_points: 16, rule: "Gauss-Legendre in ln T" },
      }),
    ).toBe("lookback T marginalised over a log-uniform prior 50 kyr–10 Myr (16-point Gauss-Legendre in ln T, likelihood space)");
  });
});
