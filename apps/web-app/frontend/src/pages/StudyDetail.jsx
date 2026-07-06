import { useEffect, useRef, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { api } from "../lib/api.js";
import Breadcrumb from "../components/Breadcrumb.jsx";
import { useConfirm } from "../components/ConfirmModal.jsx";
import Spinner from "../components/Spinner.jsx";

const fmtDate = (iso) => {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleString(undefined,
      { year: "numeric", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
  } catch { return iso; }
};

// Human-friendly duration: 1.2s · 34s · 2m 34s · 1h 12m.
const fmtSecs = (s) => {
  if (s == null) return "—";
  if (s < 60) return `${s < 10 ? s.toFixed(1) : Math.round(s)}s`;
  const m = Math.floor(s / 60), sec = Math.round(s % 60);
  if (m < 60) return `${m}m ${sec}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
};

// A labelled metadata row (mono label · value).
function Meta({ k, children }) {
  return (
    <div style={{ display: "flex", gap: 14, marginBottom: 12 }}>
      <div className="mono muted" style={{ width: 120, flexShrink: 0, fontSize: 12, textAlign: "right", paddingTop: 2 }}>{k}</div>
      <div style={{ flex: 1 }}>{children}</div>
    </div>
  );
}

// A big-number stat tile (used for the estimate + run timing).
function Stat({ label, value }) {
  return (
    <div style={{ background: "var(--surface-container)", borderRadius: 8, padding: "12px 14px" }}>
      <div className="hint mono" style={{ marginTop: 0 }}>{label}</div>
      <div style={{ fontFamily: "var(--font-display)", fontSize: 24, fontWeight: 700, color: "var(--amber-soft)", marginTop: 4 }}>{value}</div>
    </div>
  );
}

const GRID = { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: 12 };

export default function StudyDetail() {
  const { id } = useParams();
  const nav = useNavigate();
  const confirm = useConfirm();
  const [study, setStudy] = useState(null);
  const [rubric, setRubric] = useState(null);
  const [dataset, setDataset] = useState(null);
  const [timing, setTiming] = useState(null);
  const [err, setErr] = useState(null);
  const [prog, setProg] = useState(null);
  // estimate flow
  const [est, setEst] = useState(null);
  const [estimating, setEstimating] = useState(false);
  const [estErr, setEstErr] = useState(null);
  const es = useRef(null);

  const load = () => api.study(id).then((s) => {
    setStudy(s);
    if (s.rubric_id) api.rubric(s.rubric_id).then(setRubric).catch(() => setRubric(null));
    if (s.dataset_id) api.dataset(s.dataset_id).then(setDataset).catch(() => setDataset(null));
    if (s.status === "done") api.results(id).then((r) => setTiming(r.timing || null)).catch(() => setTiming(null));
    return s;
  }).catch((e) => setErr(e.message));

  const listen = () => {
    es.current?.close();
    const src = new EventSource(api.streamUrl(id));
    es.current = src;
    src.onmessage = (e) => {
      const p = JSON.parse(e.data);
      setProg(p);
      if (p.status === "done" || p.status === "failed") { src.close(); load(); }
    };
    src.onerror = () => src.close();
  };

  useEffect(() => {
    load().then((s) => { if (s?.status === "running") listen(); });
    return () => es.current?.close();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id]);

  const run = async () => {
    setErr(null);
    try { await api.runStudy(id); setProg({ phase: "answers", status: "running" }); setStudy((s) => ({ ...s, status: "running" })); listen(); }
    catch (e) { setErr(e.message); }
  };
  const estimate = async () => {
    setEstErr(null); setEstimating(true); setEst(null);
    try { setEst(await api.estimateStudy(id)); }
    catch (e) { setEstErr(e.message); }
    finally { setEstimating(false); }
  };
  const archive = () => api.archiveStudy(id).then(() => nav("/studies"));
  const restore = () => api.restoreStudy(id).then(load);
  const del = () => confirm("Permanently delete?",
    `“${study.name}” and its results/ratings will be destroyed for good — this cannot be undone.`,
    () => api.deleteStudy(id).then(() => nav("/studies")));

  const running = study?.status === "running" || prog?.status === "running";
  const judging = prog?.phase === "judging";
  const pct = prog && prog.total ? Math.round((prog.done / prog.total) * 100) : 0;
  const isDraft = study && study.status === "draft" && !running;
  const runTime = timing ? (timing.run_wall_s ?? timing.wall_s) : null;

  return (
    <div>
      <Breadcrumb items={[{ label: "Studies", to: "/studies" }, { label: study ? study.name : "…" }]} />
      {err && <div className="banner">{err}</div>}
      {!study && !err && <div className="loading">loading…</div>}

      {study && (
        <>
          <div className="page-head row-between">
            <div>
              <h1 className="page-title">{study.name}</h1>
              {study.description
                ? <p className="page-sub">{study.description}</p>
                : <p className="page-sub muted">no description</p>}
            </div>
            <span className={"badge " + study.status} style={{ alignSelf: "start" }}>{study.status}</span>
          </div>

          {/* ── Actions ─────────────────────────────────────────────────────── */}
          <div className="card" style={{ marginBottom: 16 }}>
            {isDraft && (
              <div className="row-between" style={{ flexWrap: "wrap", gap: 12 }}>
                <div className="hint" style={{ margin: 0 }}>This study hasn't been run yet — launch it to generate answers and judge them.</div>
                <button className="btn primary" onClick={run}>▶ Run study</button>
              </div>
            )}
            {running && (
              <div>
                <div className="progress"><div className={"progress-fill" + (judging ? " judging" : "")} style={{ width: `${pct}%` }} /></div>
                <div className="hint mono" style={{ marginTop: 6 }}>{judging ? "judging" : "answering"} {prog?.total ? `${prog.done}/${prog.total}` : "…"}</div>
              </div>
            )}
            {study.status === "done" && !running && (
              <div className="row-between" style={{ flexWrap: "wrap", gap: 12 }}>
                <button className="btn primary" onClick={() => nav(`/results?study=${study.id}`)}>View results →</button>
                <div style={{ display: "flex", gap: 8 }}>
                  <button className="btn" onClick={run}>Re-run</button>
                  {study.archived
                    ? <>
                        <button className="btn" onClick={restore}>Restore</button>
                        <button className="btn danger" onClick={del}>Delete forever</button>
                      </>
                    : <button className="btn" onClick={archive}>Archive</button>}
                </div>
              </div>
            )}
            {study.status === "failed" && !running && (
              <div className="row-between" style={{ flexWrap: "wrap", gap: 12 }}>
                <div className="hint" style={{ margin: 0, color: "var(--error)" }}>The last run failed. {prog?.error}</div>
                <div style={{ display: "flex", gap: 8 }}>
                  <button className="btn primary" onClick={run}>Re-run</button>
                  <button className="btn" onClick={archive}>Archive</button>
                </div>
              </div>
            )}
          </div>

          {/* ── Estimate (draft only) ───────────────────────────────────────── */}
          {isDraft && (
            <>
              <div className="section-label">Estimate</div>
              <div className="card" style={{ marginBottom: 16 }}>
                {!est && !estimating && (
                  <div className="row-between" style={{ flexWrap: "wrap", gap: 12 }}>
                    <div className="hint" style={{ margin: 0 }}>Runs one input through every configuration to project the full run's time & cost.</div>
                    <button className="btn" onClick={estimate}>Estimate time & cost</button>
                  </div>
                )}
                {estimating && <Spinner label="estimating — running one input through each configuration…" />}
                {estErr && <div className="hint" style={{ color: "var(--error)" }}>{estErr}</div>}
                {est && (
                  <div>
                    <div style={GRID}>
                      <Stat label="full run" value={`${est.estimate.total_cells} cells`} />
                      <Stat label="≈ compute time" value={fmtSecs(est.estimate.est_total_compute_s)} />
                      <Stat label="≈ cost" value={est.estimate.est_total_cost_usd != null ? `$${est.estimate.est_total_cost_usd}` : "n/a"} />
                      <Stat label="judge calls" value={est.judge_calls ?? "—"} />
                    </div>
                    <div className="hint" style={{ marginTop: 10 }}>
                      compute time is summed across cells — wall-clock is lower with parallelism; judging time isn't included.</div>
                    {est.warnings?.length > 0 && (
                      <div style={{ marginTop: 10 }}>
                        {est.warnings.map((w, i) => <div key={i} className="hint" style={{ color: "var(--amber-soft)" }}>⚠ {w}</div>)}
                      </div>
                    )}
                    <button className="btn btn-sm" style={{ marginTop: 12 }} onClick={estimate}>re-estimate</button>
                  </div>
                )}
              </div>
            </>
          )}

          {/* ── Overview / metadata ─────────────────────────────────────────── */}
          <div className="section-label">Overview</div>
          <div className="card">
            <Meta k="created">{fmtDate(study.created_at)}</Meta>
            {study.status === "done" && runTime != null && (
              <Meta k="run time"><span className="mono" style={{ color: "var(--amber-soft)" }}>{fmtSecs(runTime)}</span>
                <span className="muted mono" style={{ fontSize: 12, marginLeft: 8 }}>· full breakdown in Results</span></Meta>
            )}
            <Meta k="pipeline"><span className="mono" style={{ color: "var(--amber-soft)" }}>{study.pipeline}</span></Meta>
            <Meta k="dataset">
              {dataset
                ? <span>{dataset.name} <span className="muted mono" style={{ fontSize: 12 }}>· {dataset.items.length} questions</span></span>
                : <span className="muted">—</span>}
            </Meta>
            <Meta k="rubric">
              {rubric
                ? <span>{rubric.name} <span className="muted mono" style={{ fontSize: 12 }}>· {rubric.scale_type}</span></span>
                : <span className="muted">—</span>}
            </Meta>
            <Meta k="judge model"><span className="mono">{study.judge_model || "—"}</span></Meta>
            <Meta k="replications"><span className="mono">{study.replications}</span></Meta>
            <Meta k="concurrency"><span className="mono">{study.concurrency}</span></Meta>
            <Meta k="factors">
              <div>{study.factors.length === 0
                ? <span className="muted">—</span>
                : study.factors.map((f) => (
                  <div key={f.name} style={{ marginBottom: 6 }}>
                    <span className="mono" style={{ color: "var(--amber-soft)" }}>{f.name}</span>{" "}
                    <span className="muted mono" style={{ fontSize: 12 }}>
                      {f.levels.map(String).join(" · ")} <span style={{ opacity: 0.6 }}>({f.levels.length})</span></span>
                  </div>
                ))}</div>
            </Meta>
            <Meta k="configs">
              <span className="mono">{study.factors.reduce((a, f) => a * f.levels.length, study.factors.length ? 1 : 0)}</span>
            </Meta>
          </div>
        </>
      )}
    </div>
  );
}
