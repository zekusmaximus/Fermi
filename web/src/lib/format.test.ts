import { describe, expect, it } from "vitest";
import {
  ILLUSTRATIVE_BANNER,
  cellColor,
  decadeTicks,
  fmtLog10,
  isCleanGitSha,
  isIllustrative,
  parseSeed,
  shortSha,
  showIllustrativeBanner,
  tierClass,
  worstTier,
} from "./format";

describe("data-tier badge classification", () => {
  it("maps each tier to its own colour class", () => {
    expect(tierClass("published_catalog")).toBe("published");
    expect(tierClass("reproduced_archive")).toBe("archive");
    expect(tierClass("illustrative_calibration")).toBe("illustrative");
    expect(tierClass("synthetic_proxy")).toBe("illustrative");
    expect(tierClass(undefined)).toBe("model");
    expect(tierClass("made_up_tier")).toBe("unknown");
  });

  it("lets illustrative dominate any mixed tier set (never blends into verified)", () => {
    expect(worstTier(["published_catalog", "reproduced_archive"])).toBe("archive");
    expect(worstTier(["published_catalog", "illustrative_calibration"])).toBe("illustrative");
    expect(worstTier(["published_catalog"])).toBe("published");
    expect(worstTier([])).toBe("model");
    expect(isIllustrative(["reproduced_archive"])).toBe(false);
    expect(isIllustrative(["reproduced_archive", "synthetic_proxy"])).toBe(true);
  });
});

describe("illustrative-warning banner gating", () => {
  it("is off for the default verified headline", () => {
    expect(showIllustrativeBanner("verified_channels_only", ["published_catalog"])).toBe(false);
  });
  it("is on when the policy folds illustrative channels in", () => {
    expect(showIllustrativeBanner("including_illustrative_channels", ["published_catalog"])).toBe(true);
  });
  it("is on when the user requested it, even before the response arrives", () => {
    expect(showIllustrativeBanner("verified_channels_only", [], true)).toBe(true);
  });
  it("is on if an illustrative tier ever leaks into the reported headline", () => {
    expect(showIllustrativeBanner("verified_channels_only", ["illustrative_calibration"])).toBe(true);
  });
  it("uses the exact banner wording", () => {
    expect(ILLUSTRATIVE_BANNER).toContain("NOT AN OBSERVATIONAL BOUND");
  });
});

describe("SHA-256 truncation / display helpers", () => {
  const sha = "2da2a234ee0ccb07fb571431f5f178e362ff4d9d4f96d4ee878b921916127b7a";
  it("truncates hex hashes", () => {
    expect(shortSha(sha)).toBe("2da2a234");
    expect(shortSha(sha, 12)).toBe("2da2a234ee0c");
  });
  it("passes non-hex markers through and handles null", () => {
    expect(shortSha("dirty")).toBe("dirty");
    expect(shortSha(null)).toBe("—");
  });
  it("recognises only clean 40-hex commits", () => {
    expect(isCleanGitSha("7f22a369183364b6e3186158cd84fe8798c1900a")).toBe(true);
    expect(isCleanGitSha("dirty")).toBe(false);
    expect(isCleanGitSha(sha)).toBe(false);
  });
});

describe("numeric formatting", () => {
  it("signs log10 BFs and uses scientific notation for tiny values", () => {
    expect(fmtLog10(-8.04966e-5)).toBe("−8.050e-5");
    expect(fmtLog10(0.128762)).toBe("+0.1288");
    expect(fmtLog10(null)).toBe("—");
  });
  it("bounds seeds to uint32", () => {
    expect(parseSeed("42")).toBe(42);
    expect(parseSeed("-1")).toBeNull();
    expect(parseSeed("4294967296")).toBeNull();
    expect(parseSeed("4.2")).toBeNull();
  });
  it("colours and ticks", () => {
    expect(cellColor(Number.NaN, "prob")).toBe("transparent");
    expect(cellColor(0.1, "delta")).toContain("52, 211, 153");
    expect(cellColor(-0.1, "delta")).toContain("248, 113, 113");
    expect(decadeTicks(1.3, 800)).toEqual([1, 10, 100, 1000]);
  });
});
