// server/lib/storage.js
// =========================================================================
// Thin storage interface. Local disk today; the surface (put/read/absPath) is
// deliberately small so a later swap to S3/GCS is a new implementation of the
// same three calls, not a rewrite of callers.
//
// Images live on disk under DATA_DIR/images — NEVER in the database and NEVER
// under any web-served static root. The console streams them through an
// auth-gated route; there is no public URL.
// =========================================================================

import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Repo-root/data by default; override with DATA_DIR for a mounted volume.
export const DATA_DIR = process.env.DATA_DIR || path.resolve(__dirname, "../../data");
export const IMAGES_DIR = path.join(DATA_DIR, "images");
export const DB_PATH = process.env.DB_PATH || path.join(DATA_DIR, "derma.sqlite");

export function ensureDirs() {
  fs.mkdirSync(IMAGES_DIR, { recursive: true });
}

// Persist image bytes; returns a path RELATIVE to DATA_DIR (what we store in the
// DB) so the dataset is portable if DATA_DIR moves.
export function putImage(caseId, buffer) {
  ensureDirs();
  const rel = path.join("images", `${caseId}.jpg`);
  fs.writeFileSync(path.join(DATA_DIR, rel), buffer);
  return rel;
}

export function absPath(relPath) {
  return path.join(DATA_DIR, relPath);
}

export function readImage(relPath) {
  return fs.readFileSync(absPath(relPath));
}

export function removeImage(relPath) {
  try { fs.unlinkSync(absPath(relPath)); } catch { /* already gone */ }
}
