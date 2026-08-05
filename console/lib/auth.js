// console/lib/auth.js
// =========================================================================
// Console access gate. A single shared admin credential (CONSOLE_PASSWORD) is
// exchanged for a signed, httpOnly session cookie. The cookie also carries the
// LABELLER IDENTITY chosen at login, which is stamped on every label — that is
// what makes the multi-labeller / adjudication model work under one credential.
//
// No external dep: HMAC-signed cookie via node:crypto. Swap this module for
// real per-user accounts later without touching the routes.
// =========================================================================

import crypto from "crypto";

const COOKIE = "derma_console";
const MAX_AGE = 60 * 60 * 12; // 12h

function secret() {
  // Prefer an explicit secret; otherwise derive one from the password so
  // cookies invalidate if the password changes. Never ship a static default.
  const s = process.env.CONSOLE_SECRET || process.env.CONSOLE_PASSWORD;
  if (!s) throw new Error("CONSOLE_PASSWORD (or CONSOLE_SECRET) must be set");
  return crypto.createHash("sha256").update(s).digest();
}

function sign(payloadB64) {
  return crypto.createHmac("sha256", secret()).update(payloadB64).digest("base64url");
}

export function passwordOk(pw) {
  const expected = process.env.CONSOLE_PASSWORD || "";
  if (!expected || !pw) return false;
  const a = Buffer.from(String(pw));
  const b = Buffer.from(expected);
  return a.length === b.length && crypto.timingSafeEqual(a, b);
}

export function makeToken(labellerId) {
  const payload = { sub: labellerId, iat: Math.floor(Date.now() / 1000) };
  const body = Buffer.from(JSON.stringify(payload)).toString("base64url");
  return `${body}.${sign(body)}`;
}

export function verifyToken(token) {
  if (!token || !token.includes(".")) return null;
  const [body, sig] = token.split(".");
  const expected = sign(body);
  const a = Buffer.from(sig || ""), b = Buffer.from(expected);
  if (a.length !== b.length || !crypto.timingSafeEqual(a, b)) return null;
  try {
    const p = JSON.parse(Buffer.from(body, "base64url").toString());
    if (Math.floor(Date.now() / 1000) - p.iat > MAX_AGE) return null;
    return p;
  } catch { return null; }
}

function readCookie(req) {
  const raw = req.headers.cookie || "";
  for (const part of raw.split(";")) {
    const [k, ...v] = part.trim().split("=");
    if (k === COOKIE) return decodeURIComponent(v.join("="));
  }
  return null;
}

export function setSession(res, labellerId) {
  const token = makeToken(labellerId);
  const attrs = [`${COOKIE}=${encodeURIComponent(token)}`, "HttpOnly", "SameSite=Strict", "Path=/", `Max-Age=${MAX_AGE}`];
  if (process.env.NODE_ENV === "production") attrs.push("Secure");
  res.setHeader("Set-Cookie", attrs.join("; "));
}

export function clearSession(res) {
  res.setHeader("Set-Cookie", `${COOKIE}=; HttpOnly; SameSite=Strict; Path=/; Max-Age=0`);
}

// Express middleware: 401 unless a valid session cookie is present. Attaches
// req.labellerId for downstream attribution.
export function requireAuth(req, res, next) {
  const p = verifyToken(readCookie(req));
  if (!p) return res.status(401).json({ error: "unauthorized" });
  req.labellerId = p.sub;
  next();
}
