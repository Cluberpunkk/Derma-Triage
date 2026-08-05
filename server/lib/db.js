// server/lib/db.js
// =========================================================================
// SQLite access + schema. Shared substrate for the public ingestion endpoint
// (writes cases) and the labelling console (writes labels, reads everything).
// Images live on disk (see storage.js); the DB only holds the relative path.
//
// Multi-labeller by design: `labels` is one-to-many with `cases`, keyed by
// labeller_id, with a unique (case_id, labeller_id) so each labeller has one
// current label per case (re-submitting updates it). Adjudication picks a gold
// label per case in `adjudications`.
// =========================================================================

import Database from "better-sqlite3";
import crypto from "crypto";
import { DB_PATH, ensureDirs } from "./storage.js";

let _db = null;

export function db() {
  if (_db) return _db;
  ensureDirs();
  _db = new Database(DB_PATH);
  _db.pragma("journal_mode = WAL");
  _db.pragma("foreign_keys = ON");
  migrate(_db);
  return _db;
}

function migrate(d) {
  d.exec(`
    CREATE TABLE IF NOT EXISTS cases (
      case_id        TEXT PRIMARY KEY,
      created_at     TEXT NOT NULL,
      image_path     TEXT NOT NULL,
      body_location  TEXT,
      duration       TEXT,
      symptoms       TEXT,           -- JSON array (patient-reported)
      recent_change  TEXT,
      prior_cream    TEXT,
      steroid        TEXT,
      skin_tone_self TEXT,           -- self-reported; NEVER ground truth
      consent        INTEGER NOT NULL DEFAULT 0,
      consent_ts     TEXT,
      consent_version TEXT,
      exif_stripped  INTEGER NOT NULL DEFAULT 0,
      face_blurred   INTEGER NOT NULL DEFAULT 0,
      withdrawn      INTEGER NOT NULL DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS labels (
      label_id          TEXT PRIMARY KEY,
      case_id           TEXT NOT NULL REFERENCES cases(case_id) ON DELETE CASCADE,
      labeller_id       TEXT NOT NULL,
      confirmed_diagnosis TEXT,
      confirmation_method TEXT,
      fitzpatrick       TEXT,
      monk              INTEGER,
      body_site         TEXT,
      prior_steroid_use TEXT,
      image_gradable    TEXT,
      gradable_reason   TEXT,
      needs_confirmation INTEGER NOT NULL DEFAULT 0,
      notes             TEXT,
      created_at        TEXT NOT NULL,
      updated_at        TEXT NOT NULL,
      UNIQUE(case_id, labeller_id)
    );

    -- Gold label chosen per case when labellers disagree (or to bless a single label).
    CREATE TABLE IF NOT EXISTS adjudications (
      case_id      TEXT PRIMARY KEY REFERENCES cases(case_id) ON DELETE CASCADE,
      label_id     TEXT NOT NULL REFERENCES labels(label_id) ON DELETE CASCADE,
      adjudicator  TEXT NOT NULL,
      created_at   TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_labels_case ON labels(case_id);
  `);
}

export function newId(prefix) {
  return `${prefix}_${crypto.randomBytes(9).toString("hex")}`;
}

// ---- cases --------------------------------------------------------------

export function insertCase(row) {
  db().prepare(`
    INSERT INTO cases (case_id, created_at, image_path, body_location, duration,
      symptoms, recent_change, prior_cream, steroid, skin_tone_self,
      consent, consent_ts, consent_version, exif_stripped, face_blurred)
    VALUES (@case_id, @created_at, @image_path, @body_location, @duration,
      @symptoms, @recent_change, @prior_cream, @steroid, @skin_tone_self,
      @consent, @consent_ts, @consent_version, @exif_stripped, @face_blurred)
  `).run(row);
}

export function getCase(caseId) {
  return db().prepare(`SELECT * FROM cases WHERE case_id = ?`).get(caseId);
}

export function withdrawCase(caseId) {
  return db().prepare(`UPDATE cases SET withdrawn = 1 WHERE case_id = ?`).run(caseId);
}

// ---- labels -------------------------------------------------------------

export function getLabels(caseId) {
  return db().prepare(`SELECT * FROM labels WHERE case_id = ? ORDER BY updated_at DESC`).all(caseId);
}

export function upsertLabel(caseId, labellerId, l) {
  const now = new Date().toISOString();
  const existing = db().prepare(`SELECT label_id, created_at FROM labels WHERE case_id = ? AND labeller_id = ?`).get(caseId, labellerId);
  const row = {
    label_id: existing ? existing.label_id : newId("lbl"),
    case_id: caseId,
    labeller_id: labellerId,
    confirmed_diagnosis: l.confirmed_diagnosis ?? null,
    confirmation_method: l.confirmation_method ?? null,
    fitzpatrick: l.fitzpatrick ?? null,
    monk: l.monk != null ? Number(l.monk) : null,
    body_site: l.body_site ?? null,
    prior_steroid_use: l.prior_steroid_use ?? null,
    image_gradable: l.image_gradable ?? null,
    gradable_reason: l.gradable_reason ?? null,
    needs_confirmation: l.needs_confirmation ? 1 : 0,
    notes: l.notes ?? null,
    created_at: existing ? existing.created_at : now,
    updated_at: now,
  };
  db().prepare(`
    INSERT INTO labels (label_id, case_id, labeller_id, confirmed_diagnosis, confirmation_method,
      fitzpatrick, monk, body_site, prior_steroid_use, image_gradable, gradable_reason,
      needs_confirmation, notes, created_at, updated_at)
    VALUES (@label_id, @case_id, @labeller_id, @confirmed_diagnosis, @confirmation_method,
      @fitzpatrick, @monk, @body_site, @prior_steroid_use, @image_gradable, @gradable_reason,
      @needs_confirmation, @notes, @created_at, @updated_at)
    ON CONFLICT(case_id, labeller_id) DO UPDATE SET
      confirmed_diagnosis=excluded.confirmed_diagnosis,
      confirmation_method=excluded.confirmation_method,
      fitzpatrick=excluded.fitzpatrick, monk=excluded.monk, body_site=excluded.body_site,
      prior_steroid_use=excluded.prior_steroid_use, image_gradable=excluded.image_gradable,
      gradable_reason=excluded.gradable_reason, needs_confirmation=excluded.needs_confirmation,
      notes=excluded.notes, updated_at=excluded.updated_at
  `).run(row);
  return row;
}

export function setAdjudication(caseId, labelId, adjudicator) {
  const now = new Date().toISOString();
  db().prepare(`
    INSERT INTO adjudications (case_id, label_id, adjudicator, created_at)
    VALUES (?, ?, ?, ?)
    ON CONFLICT(case_id) DO UPDATE SET label_id=excluded.label_id,
      adjudicator=excluded.adjudicator, created_at=excluded.created_at
  `).run(caseId, labelId, adjudicator, now);
}

export function getAdjudication(caseId) {
  return db().prepare(`SELECT * FROM adjudications WHERE case_id = ?`).get(caseId);
}

// ---- queue / listing ----------------------------------------------------

// Cases for the review queue. `filter`: 'unlabelled' (no label by this labeller),
// 'needs_confirmation', 'conflicts' (2+ labels), or 'all'. Withdrawn excluded.
export function queue(labellerId, filter = "unlabelled", limit = 200) {
  const rows = db().prepare(`
    SELECT c.*,
      (SELECT COUNT(*) FROM labels l WHERE l.case_id = c.case_id) AS label_count,
      (SELECT COUNT(*) FROM labels l WHERE l.case_id = c.case_id AND l.labeller_id = ?) AS mine,
      (SELECT COUNT(*) FROM labels l WHERE l.case_id = c.case_id AND l.needs_confirmation = 1) AS needs_conf,
      (SELECT COUNT(DISTINCT confirmed_diagnosis) FROM labels l WHERE l.case_id = c.case_id AND confirmed_diagnosis IS NOT NULL) AS distinct_dx
    FROM cases c
    WHERE c.withdrawn = 0 AND c.consent = 1
    ORDER BY c.created_at ASC
  `).all(labellerId);

  return rows.filter((r) => {
    if (filter === "unlabelled") return r.mine === 0;
    if (filter === "needs_confirmation") return r.needs_conf > 0;
    if (filter === "conflicts") return r.distinct_dx > 1;
    return true;
  }).slice(0, limit);
}
