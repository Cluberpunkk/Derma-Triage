import React, { useCallback, useEffect, useRef, useState } from "react";

// Small fetch wrapper: cookies are same-origin; 401 bubbles up to force re-login.
async function api(path, opts = {}) {
  const res = await fetch(path, {
    credentials: "same-origin",
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (res.status === 401) { const e = new Error("unauthorized"); e.unauth = true; throw e; }
  return res;
}

const EMPTY = {
  confirmed_diagnosis: "", confirmation_method: "", fitzpatrick: "", monk: "",
  body_site: "", prior_steroid_use: "", image_gradable: "", gradable_reason: "",
  needs_confirmation: false, notes: "",
};

export default function Console() {
  const [me, setMe] = useState(null);
  const [booting, setBooting] = useState(true);
  const [tab, setTab] = useState("queue");
  const [vocab, setVocab] = useState(null);

  useEffect(() => {
    api("/console-api/me")
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then((d) => { setMe(d.labellerId); return api("/console-api/vocab").then((r) => r.json()).then(setVocab); })
      .catch(() => {})
      .finally(() => setBooting(false));
  }, []);

  const onLogout = async () => { await api("/console-api/logout", { method: "POST" }); setMe(null); };
  const onLogin = async (labellerId) => { setMe(labellerId); const v = await api("/console-api/vocab").then((r) => r.json()); setVocab(v); };

  if (booting) return null;
  if (!me) return <Login onLogin={onLogin} />;

  return (
    <>
      <div className="topbar">
        <div>
          <span className="brand">Derma<b>Triage</b> Labelling</span>
          <span className="tag">Expert console</span>
        </div>
        <div className="tabs">
          <button className={tab === "queue" ? "on" : ""} onClick={() => setTab("queue")}>Review queue</button>
          <button className={tab === "dashboard" ? "on" : ""} onClick={() => setTab("dashboard")}>Data quality</button>
        </div>
        <div className="who">
          <span>Labelling as <b>{me}</b></span>
          <button onClick={onLogout}>Sign out</button>
        </div>
      </div>
      {vocab && (tab === "queue" ? <Workspace vocab={vocab} /> : <Dashboard />)}
    </>
  );
}

// ------------------------------------------------------------------ Login
function Login({ onLogin }) {
  const [labeller, setLabeller] = useState("");
  const [password, setPassword] = useState("");
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);

  const submit = async (e) => {
    e.preventDefault();
    setBusy(true); setErr("");
    try {
      const r = await api("/console-api/login", { method: "POST", body: JSON.stringify({ labeller, password }) });
      const d = await r.json();
      if (!r.ok) { setErr(d.error || "Login failed"); return; }
      onLogin(d.labellerId);
    } catch { setErr("Login failed"); }
    finally { setBusy(false); }
  };

  return (
    <form className="login-wrap" onSubmit={submit}>
      <h1>Labelling console</h1>
      <p>Dermatologist access only. Your labeller name is stamped on every label you confirm.</p>
      <div className="fld">
        <label htmlFor="lb">Labeller name</label>
        <input id="lb" value={labeller} onChange={(e) => setLabeller(e.target.value)} placeholder="e.g. Dr A. Rao" autoFocus />
      </div>
      <div className="fld">
        <label htmlFor="pw">Console password</label>
        <input id="pw" type="password" value={password} onChange={(e) => setPassword(e.target.value)} />
      </div>
      <button className="primary" disabled={busy || !labeller || !password}>Enter console</button>
      {err && <div className="err">{err}</div>}
    </form>
  );
}

// -------------------------------------------------------------- Workspace
const FILTERS = [
  { key: "unlabelled", label: "Unlabelled" },
  { key: "needs_confirmation", label: "Needs confirmation" },
  { key: "conflicts", label: "Conflicts" },
  { key: "all", label: "All" },
];

function Workspace({ vocab }) {
  const [filter, setFilter] = useState("unlabelled");
  const [cases, setCases] = useState([]);
  const [idx, setIdx] = useState(0);
  const [detail, setDetail] = useState(null);
  const [form, setForm] = useState(EMPTY);
  const [save, setSave] = useState({ state: "idle", missing: [], confirmed: false });
  const [zoom, setZoom] = useState(false);

  const justLoaded = useRef(false);
  const saveTimer = useRef(0);
  const formRef = useRef(form);
  formRef.current = form;

  const loadQueue = useCallback((keepId) => {
    api(`/console-api/queue?filter=${filter}`).then((r) => r.json()).then((d) => {
      setCases(d.cases);
      if (keepId) { const i = d.cases.findIndex((c) => c.case_id === keepId); setIdx(i >= 0 ? i : 0); }
      else setIdx(0);
    });
  }, [filter]);

  useEffect(() => { loadQueue(); }, [loadQueue]);

  const current = cases[idx];

  // Load case detail + prefetch the next image.
  useEffect(() => {
    if (!current) { setDetail(null); return; }
    let live = true;
    api(`/console-api/case/${current.case_id}`).then((r) => r.json()).then((d) => {
      if (!live) return;
      setDetail(d);
      justLoaded.current = true;
      setForm(d.mine ? mineToForm(d.mine) : EMPTY);
      setSave({ state: "idle", missing: [], confirmed: false });
      setZoom(false);
    });
    const next = cases[idx + 1];
    if (next) { const im = new Image(); im.src = `/console-api/image/${next.case_id}`; }
    return () => { live = false; };
  }, [current, idx, cases]);

  // Autosave (debounced) whenever the form changes from user input.
  useEffect(() => {
    if (justLoaded.current) { justLoaded.current = false; return; }
    if (!current) return;
    setSave((s) => ({ ...s, state: "saving" }));
    clearTimeout(saveTimer.current);
    saveTimer.current = setTimeout(() => doSave(false), 600);
    return () => clearTimeout(saveTimer.current);
  }, [form]); // eslint-disable-line

  const doSave = async (advance) => {
    if (!current) return;
    const body = { ...formRef.current, monk: formRef.current.monk === "" ? null : Number(formRef.current.monk) };
    const r = await api(`/console-api/case/${current.case_id}/label`, { method: "POST", body: JSON.stringify(body) });
    const d = await r.json();
    setSave({ state: "saved", missing: d.errors || [], confirmed: d.confirmed });
    if (advance) nextCase();
    else loadQueueBadgesSoon();
  };

  const badgeTimer = useRef(0);
  const loadQueueBadgesSoon = () => { clearTimeout(badgeTimer.current); badgeTimer.current = setTimeout(() => loadQueue(current?.case_id), 400); };

  const nextCase = () => setIdx((i) => Math.min(i + 1, cases.length - 1));
  const prevCase = () => setIdx((i) => Math.max(i - 1, 0));
  const set = (patch) => setForm((f) => ({ ...f, ...patch }));

  // -------- hotkeys --------
  useEffect(() => {
    const onKey = (e) => {
      const t = e.target;
      const typing = t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT");
      if (typing) { if (e.key === "Escape") t.blur(); return; }
      if (e.key === "]" || e.key === "ArrowRight") { e.preventDefault(); nextCase(); }
      else if (e.key === "[" || e.key === "ArrowLeft") { e.preventDefault(); prevCase(); }
      else if (e.key === "Enter") { e.preventDefault(); clearTimeout(saveTimer.current); doSave(true); }
      else if (e.key === "g") { set({ image_gradable: formRef.current.image_gradable === "yes" ? "no" : "yes" }); }
      else if (e.key === "f") { set({ needs_confirmation: !formRef.current.needs_confirmation }); }
      else if (/^[0-9]$/.test(e.key)) {
        const n = e.key === "0" ? 9 : Number(e.key) - 1;
        const d = vocab.DIAGNOSES[n];
        if (d) set({ confirmed_diagnosis: d.key });
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [cases, vocab]); // eslint-disable-line

  return (
    <div className="layout">
      {/* queue */}
      <div className="qpane">
        <div className="qfilters">
          {FILTERS.map((f) => (
            <button key={f.key} className={filter === f.key ? "on" : ""} onClick={() => setFilter(f.key)}>{f.label}</button>
          ))}
        </div>
        {cases.length === 0 && <div className="empty">Queue empty</div>}
        {cases.map((c, i) => (
          <div key={c.case_id} className={"qitem" + (i === idx ? " sel" : "")} onClick={() => setIdx(i)}>
            <span className="cid">{c.case_id.slice(0, 12)}</span>
            <span className="badges">
              {c.mine > 0 && <span className="b mine">mine</span>}
              {c.needs_conf > 0 && <span className="b conf">conf</span>}
              {c.distinct_dx > 1 && <span className="b conflict">≠</span>}
            </span>
          </div>
        ))}
      </div>

      {/* viewer */}
      <div className="viewer">
        {!current && <div className="empty">No case selected</div>}
        {current && detail && (
          <>
            <div className="imgwrap">
              <img
                src={`/console-api/image/${current.case_id}`}
                alt="case"
                className={zoom ? "zoom" : ""}
                style={zoom ? { transform: "scale(2)" } : undefined}
                onClick={() => setZoom((z) => !z)}
              />
            </div>
            <PatientMeta c={detail.case} />
          </>
        )}
      </div>

      {/* label form */}
      <div className="form">
        {current && detail ? (
          <>
            <div className="savebar">
              <span className={"state " + (save.state === "saving" ? "" : save.confirmed ? "ok" : "missing")}>
                {save.state === "saving" ? "Saving…"
                  : save.state === "idle" ? (detail.mine ? "Loaded" : "New case")
                  : save.confirmed ? "Saved · confirmed ✓" : `Saved · ${save.missing.length} to fix`}
              </span>
              <button className="opt" onClick={() => { clearTimeout(saveTimer.current); doSave(true); }}>Save + next ⏎</button>
            </div>

            {detail.labels.length > 1 && <Adjudication detail={detail} vocab={vocab} onDone={() => loadQueue(current.case_id)} caseId={current.case_id} />}

            <Group label="Confirmed diagnosis" req>
              <div className="opts">
                {vocab.DIAGNOSES.map((d, i) => (
                  <button key={d.key} className={"opt dx" + (form.confirmed_diagnosis === d.key ? " on" : "")} onClick={() => set({ confirmed_diagnosis: d.key })}>
                    {i < 10 && <span className="hk">{(i + 1) % 10}</span>}{d.label}
                  </button>
                ))}
              </div>
            </Group>

            <Group label="Confirmation method" req>
              <Opts items={vocab.CONFIRMATION_METHODS} value={form.confirmation_method} onPick={(v) => set({ confirmation_method: v })} />
            </Group>

            <Group label="Fitzpatrick" req>
              <Opts items={vocab.FITZPATRICK.map((f) => ({ key: f, label: f }))} value={form.fitzpatrick} onPick={(v) => set({ fitzpatrick: v })} />
            </Group>

            <Group label="Monk skin tone" req>
              <Opts items={vocab.MONK.map((m) => ({ key: m, label: String(m) }))} value={form.monk} onPick={(v) => set({ monk: v })} />
            </Group>

            <Group label="Body site">
              <select value={form.body_site} onChange={(e) => set({ body_site: e.target.value })}>
                <option value="">—</option>
                {vocab.BODY_SITES.map((b) => <option key={b} value={b}>{b}</option>)}
              </select>
            </Group>

            <Group label="Prior steroid use">
              <Opts items={vocab.PRIOR_STEROID.map((p) => ({ key: p, label: p }))} value={form.prior_steroid_use} onPick={(v) => set({ prior_steroid_use: v })} />
            </Group>

            <Group label="Image gradable" req>
              <Opts items={[{ key: "yes", label: "Yes (g)" }, { key: "no", label: "No (g)" }]} value={form.image_gradable} onPick={(v) => set({ image_gradable: v })} />
              {form.image_gradable === "no" && (
                <input className="search" style={{ marginTop: 6 }} placeholder="Why not gradable? (required)" value={form.gradable_reason} onChange={(e) => set({ gradable_reason: e.target.value })} />
              )}
            </Group>

            <Group label="">
              <label className="chk">
                <input type="checkbox" checked={form.needs_confirmation} onChange={(e) => set({ needs_confirmation: e.target.checked })} />
                Needs confirmation — chase KOH/culture (f)
              </label>
            </Group>

            <Group label="Notes">
              <textarea value={form.notes} onChange={(e) => set({ notes: e.target.value })} placeholder="Free-text notes" />
            </Group>

            <div className="hotkeys">
              <code>1–9,0</code> diagnosis · <code>g</code> gradable · <code>f</code> flag · <code>⏎</code> save+next · <code>[ ]</code> / <code>← →</code> prev/next
            </div>
          </>
        ) : <div className="empty">Select a case to label</div>}
      </div>
    </div>
  );
}

function Group({ label, req, children }) {
  return (
    <div className="grp">
      {label !== "" && <div className="lab"><span>{label}</span>{req && <span className="req">required</span>}</div>}
      {children}
    </div>
  );
}

function Opts({ items, value, onPick }) {
  return (
    <div className="opts">
      {items.map((it) => (
        <button key={it.key} className={"opt" + (String(value) === String(it.key) ? " on" : "")} onClick={() => onPick(it.key)}>{it.label}</button>
      ))}
    </div>
  );
}

function PatientMeta({ c }) {
  const row = (k, v) => <><dt>{k}</dt><dd>{v || "—"}</dd></>;
  const syms = Array.isArray(c.symptoms) ? c.symptoms.join(", ") : c.symptoms;
  return (
    <div className="meta">
      <h3>Patient-reported (context only — not ground truth)</h3>
      <dl>
        {row("Case", c.case_id)}
        {row("Captured", c.created_at?.replace("T", " ").slice(0, 16))}
        {row("Body location", c.body_location)}
        {row("Duration", c.duration)}
        {row("Symptoms", syms)}
        {row("Recent change", c.recent_change)}
        {row("Prior cream", c.prior_cream)}
        {row("Steroid cream", c.steroid)}
        {row("Self-reported tone", c.skin_tone_self)}
        {row("Face blurred", c.face_blurred ? "yes" : "no")}
      </dl>
      <div className="selfnote">These are the patient's own answers. Confirm skin tone and steroid use yourself — self-report is never treated as ground truth.</div>
    </div>
  );
}

function Adjudication({ detail, vocab, onDone, caseId }) {
  const dxLabel = (k) => (vocab.DIAGNOSES.find((d) => d.key === k) || {}).label || k || "—";
  const gold = detail.adjudication?.label_id;
  const pick = async (label_id) => {
    await api(`/console-api/case/${caseId}/adjudicate`, { method: "POST", body: JSON.stringify({ label_id }) });
    onDone();
  };
  return (
    <div className="adj">
      <h4>Multiple labels — choose the gold label</h4>
      {detail.labels.map((l) => (
        <div key={l.label_id} className={"cand" + (l.label_id === gold ? " gold" : "")}>
          <span><b>{l.labeller_id}</b>: {dxLabel(l.confirmed_diagnosis)} · {l.confirmation_method || "—"}</span>
          {l.label_id === gold ? <span>gold ★</span> : <button onClick={() => pick(l.label_id)}>Set gold</button>}
        </div>
      ))}
    </div>
  );
}

function mineToForm(m) {
  return {
    confirmed_diagnosis: m.confirmed_diagnosis || "", confirmation_method: m.confirmation_method || "",
    fitzpatrick: m.fitzpatrick || "", monk: m.monk ?? "", body_site: m.body_site || "",
    prior_steroid_use: m.prior_steroid_use || "", image_gradable: m.image_gradable || "",
    gradable_reason: m.gradable_reason || "", needs_confirmation: !!m.needs_confirmation, notes: m.notes || "",
  };
}

// -------------------------------------------------------------- Dashboard
function Dashboard() {
  const [d, setD] = useState(null);
  const [exp, setExp] = useState(null);
  const [exporting, setExporting] = useState(false);

  useEffect(() => { api("/console-api/dashboard").then((r) => r.json()).then(setD); }, []);
  const runExport = async () => {
    setExporting(true);
    try { const r = await api("/console-api/export", { method: "POST" }); setExp(await r.json()); }
    finally { setExporting(false); }
  };

  if (!d) return <div className="empty">Loading…</div>;
  const maxDx = Math.max(1, ...d.by_diagnosis.map((x) => x.count));
  const maxM = Math.max(1, ...d.by_method.map((x) => x.count));
  const bands = ["I–II", "III–IV", "V–VI"];

  return (
    <div className="dash">
      <div className="kpis">
        <Kpi n={d.totals.cases} k="Consented cases" />
        <Kpi n={d.totals.labelled} k="Labelled" />
        <Kpi n={d.totals.gold} k="Gold (export-ready)" />
        <Kpi n={d.totals.unlabelled} k="Unlabelled" />
        <Kpi n={d.totals.gradable_pct + "%"} k="Gradable (of gold)" />
        <Kpi n={d.totals.needs_confirmation} k="Flagged for confirmation" />
      </div>

      <div className="panel">
        <h3>Cases per confirmed diagnosis (gold labels)</h3>
        {d.by_diagnosis.filter((x) => x.count > 0).length === 0 && <div className="empty">No gold labels yet.</div>}
        {d.by_diagnosis.filter((x) => x.count > 0).map((x) => (
          <div className="bar" key={x.key}><span className="bl">{x.label}</span><span className="bt"><span className="bf" style={{ width: `${(x.count / maxDx) * 100}%` }} /></span><span className="bn">{x.count}</span></div>
        ))}
      </div>

      <div className="panel">
        <h3>Skin-tone coverage (Fitzpatrick band × diagnosis)</h3>
        <p style={{ fontSize: 12, color: "var(--mut)", marginTop: -4 }}>Red cells (&lt;5) flag under-represented dark-skin conditions for the held-out test set.</p>
        <table className="matrix">
          <thead><tr><th style={{ textAlign: "left" }}>Diagnosis</th>{bands.map((b) => <th key={b}>{b}</th>)}</tr></thead>
          <tbody>
            {d.coverage_matrix.filter((r) => bands.some((b) => r.bands[b] > 0)).map((r) => (
              <tr key={r.key}>
                <td className="name">{r.label}</td>
                {bands.map((b) => <td key={b} className={r.bands[b] < 5 ? "low" : ""}>{r.bands[b]}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="panel">
        <h3>Confirmation-method breakdown</h3>
        {d.by_method.map((x) => (
          <div className="bar" key={x.key}><span className="bl">{x.label}</span><span className="bt"><span className="bf" style={{ width: `${(x.count / maxM) * 100}%` }} /></span><span className="bn">{x.count}</span></div>
        ))}
      </div>

      {d.gaps.length > 0 && (
        <div className="gaps">
          <h3>Coverage gaps to chase</h3>
          <ul>{d.gaps.map((g, i) => <li key={i}>{g}</li>)}</ul>
        </div>
      )}

      <div className="panel">
        <h3>Export ML-ready dataset</h3>
        <p style={{ fontSize: 12.5, color: "var(--mut)", marginTop: -4 }}>Writes manifest.csv + manifest.json and foldered images. Splits are stratified by skin tone × diagnosis with a fixed seed (reproducible). Only dermatologist-confirmed, gradable cases enter train/val/test.</p>
        <button className="exportbtn" onClick={runExport} disabled={exporting}>{exporting ? "Exporting…" : "Run export"}</button>
        {exp && exp.ok && <div className="exres">Exported {exp.total} rows → train {exp.splits.train} · val {exp.splits.val} · test {exp.splits.test} · excluded {exp.splits.excluded_nongradable}<br />{exp.outDir}</div>}
        {exp && !exp.ok && <div className="err">Export failed.</div>}
      </div>
    </div>
  );
}

function Kpi({ n, k }) { return <div className="kpi"><div className="n">{n}</div><div className="k">{k}</div></div>; }
