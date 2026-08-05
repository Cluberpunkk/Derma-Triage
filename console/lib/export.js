// console/lib/export.js
// =========================================================================
// ML-ready dataset export. Produces, under an output dir:
//   manifest.csv, manifest.json  — one row per gold-labelled case
//   images/<split>/<diagnosis>/<case_id>.jpg  — foldered for training loaders
//
// Splits are STRATIFIED by (fitzpatrick × diagnosis) and REPRODUCIBLE via a
// fixed seed, so the held-out test set carries Fitzpatrick V–VI representation
// proportional to what's collected. Only gradable=yes rows enter train/val/test;
// non-gradable gold rows are kept in the manifest as split="excluded_nongradable".
// =========================================================================

import fs from "fs";
import path from "path";
import { db } from "../../server/lib/db.js";
import { absPath } from "../../server/lib/storage.js";
import { goldLabel } from "./gold.js";

const SEED = 42;                 // fixed → reproducible splits
const TARGET = { train: 0.70, val: 0.15, test: 0.15 };
const RATIOS = { val: TARGET.val, test: TARGET.test }; // reported in manifest.json

// Deterministic PRNG (mulberry32) — no Math.random, so exports are reproducible.
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function seededShuffle(arr, rnd) {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

// Gather every case that has a gold label. Returns manifest rows (pre-split).
export function goldRows() {
  const cases = db().prepare(`SELECT * FROM cases WHERE withdrawn = 0 AND consent = 1 ORDER BY case_id ASC`).all();
  const rows = [];
  for (const c of cases) {
    const g = goldLabel(c.case_id);
    if (!g) continue;
    const l = g.label;
    rows.push({
      case_id: c.case_id,
      image_path: c.image_path,
      confirmed_diagnosis: l.confirmed_diagnosis,
      confirmation_method: l.confirmation_method,
      fitzpatrick: l.fitzpatrick,
      monk: l.monk,
      body_site: l.body_site || "",
      prior_steroid_use: l.prior_steroid_use || "unknown",
      gradable: l.image_gradable,
      resolution: g.resolution,
    });
  }
  return rows;
}

// Assign train/val/test to gradable rows, stratified by fitzpatrick × diagnosis.
// Uses a global largest-debt allocator: for each case we pick the split whose
// running share is furthest below its target. This keeps GLOBAL ratios near
// 70/15/15 even when most strata hold only 1–2 cases (naive per-stratum rounding
// inflates whichever split "wins" tiny strata) while still spreading each
// stratum across splits. Deterministic given the fixed seed + sorted iteration.
export function assignSplits(rows) {
  const gradable = rows.filter((r) => r.gradable === "yes");
  const strata = new Map();
  for (const r of gradable) {
    const key = `${r.fitzpatrick}|${r.confirmed_diagnosis}`;
    if (!strata.has(key)) strata.set(key, []);
    strata.get(key).push(r);
  }

  const split = new Map();                 // case_id -> split
  const count = { train: 0, val: 0, test: 0 };
  let total = 0;
  for (const key of [...strata.keys()].sort()) {          // sorted keys → deterministic
    const members = strata.get(key).slice().sort((a, b) => a.case_id.localeCompare(b.case_id));
    const shuffled = seededShuffle(members, mulberry32(SEED ^ hash(key)));
    for (const r of shuffled) {
      total++;
      let best = "train", bestDebt = -Infinity;
      for (const s of ["train", "val", "test"]) {
        const debt = TARGET[s] * total - count[s];        // how far below target this split is
        if (debt > bestDebt) { bestDebt = debt; best = s; }
      }
      count[best]++;
      split.set(r.case_id, best);
    }
  }

  for (const r of rows) {
    r.split = r.gradable === "yes" ? split.get(r.case_id) : "excluded_nongradable";
  }
  return rows;
}

function hash(s) {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
  return h >>> 0;
}

const COLS = ["case_id", "image_path", "confirmed_diagnosis", "confirmation_method",
  "fitzpatrick", "monk", "body_site", "prior_steroid_use", "gradable", "split", "resolution"];

function toCsv(rows) {
  const esc = (v) => {
    const s = v == null ? "" : String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  return [COLS.join(","), ...rows.map((r) => COLS.map((c) => esc(r[c])).join(","))].join("\n");
}

// Write the full export to `outDir`. Returns a summary.
export function writeExport(outDir) {
  const rows = assignSplits(goldRows());
  fs.mkdirSync(outDir, { recursive: true });
  fs.writeFileSync(path.join(outDir, "manifest.csv"), toCsv(rows));
  fs.writeFileSync(path.join(outDir, "manifest.json"), JSON.stringify({ seed: SEED, ratios: RATIOS, count: rows.length, rows }, null, 2));

  const counts = { train: 0, val: 0, test: 0, excluded_nongradable: 0 };
  for (const r of rows) {
    counts[r.split] = (counts[r.split] || 0) + 1;
    const dest = path.join(outDir, "images", r.split, r.confirmed_diagnosis);
    fs.mkdirSync(dest, { recursive: true });
    try { fs.copyFileSync(absPath(r.image_path), path.join(dest, `${r.case_id}.jpg`)); }
    catch { /* image missing; manifest row still recorded */ }
  }
  return { outDir, total: rows.length, splits: counts };
}
