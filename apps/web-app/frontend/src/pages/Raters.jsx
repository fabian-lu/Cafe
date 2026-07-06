import { useEffect, useState } from "react";
import { api } from "../lib/api.js";

const band = (a) => (a == null ? "muted" : a >= 0.8 ? "green" : a >= 0.667 ? "amber" : "error");

function AlphaCard({ title, sub, a }) {
  const val = a?.alpha;
  const color = { green: "var(--green)", amber: "var(--amber-soft)", error: "var(--error)", muted: "var(--outline)" }[band(val)];
  return (
    <div className="card" style={{ height: "100%", marginTop: 0, display: "flex", flexDirection: "column" }}>
      <div className="hint mono" style={{ marginBottom: 2 }}>{title}</div>
      <div className="muted" style={{ fontSize: 12, marginBottom: 10 }}>{sub}</div>
      <div style={{ marginTop: "auto" }}>
        <div style={{ fontFamily: "var(--font-display)", fontSize: 40, fontWeight: 700, color, lineHeight: 1 }}>
          {val == null ? "—" : val}</div>
        <div className="hint">{a ? `${a.interpret} · ${a.n_units} shared answers` : "not enough data yet"}</div>
      </div>
    </div>
  );
}

// A compact view of the rubric the judge used — so a human grades on the SAME scale.
function RubricPanel({ rubric, judgeModel }) {
  if (!rubric) return null;
  return (
    <div className="card" style={{ marginBottom: 16, background: "var(--surface-container)" }}>
      <div className="row-between" style={{ marginBottom: 8 }}>
        <div className="hint mono">the judge's rubric — score on this exact scale</div>
        {judgeModel && <div className="muted mono" style={{ fontSize: 11 }}>judge: {judgeModel}</div>}
      </div>
      <div style={{ fontWeight: 600, marginBottom: 4 }}>{rubric.name}
        <span className="muted mono" style={{ fontSize: 11, marginLeft: 8 }}>{rubric.scale_type}</span></div>
      {rubric.instruction && <div className="muted" style={{ fontSize: 13, marginBottom: 10 }}>{rubric.instruction}</div>}
      <div>{rubric.levels.map((l) => (
        <div key={l.value} style={{ marginBottom: 3, fontSize: 13 }}>
          <span className="mono" style={{ color: "var(--amber-soft)" }}>{l.value}</span>{" "}
          <b>{l.label}</b>{l.description ? <span className="muted"> — {l.description}</span> : null}
        </div>
      ))}</div>
    </div>
  );
}

export default function Raters() {
  const [studies, setStudies] = useState([]);
  const [sel, setSel] = useState("");
  const [raters, setRaters] = useState([]);
  const [rel, setRel] = useState(null);
  // rating flow
  const [rater, setRater] = useState("");
  const [sheet, setSheet] = useState(null);   // { rubric, judge_model, items }
  const [scores, setScores] = useState({});
  const [msg, setMsg] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => { api.studies().then((all) => setStudies(all.filter((s) => s.status === "done"))); }, []);

  const refresh = (id) => {
    api.studyRaters(id).then(setRaters).catch(() => setRaters([]));
    api.reliability(id).then(setRel).catch(() => setRel(null));
  };
  const pick = (id) => {
    setSel(id); setSheet(null); setScores({}); setRater(""); setMsg(null); setErr(null);
    setRel(null); setRaters([]);
    if (id) refresh(id);
  };

  const start = () => {
    setErr(null);
    if (!rater) { setErr("enter a rater name"); return; }
    api.ratingSheet(sel).then((s) => { setSheet(s); setScores({}); }).catch((e) => setErr(e.message));
  };
  const submit = async () => {
    setErr(null); setMsg(null);
    try {
      const r = await api.submitHumanRatings(sel, { rater, scores });
      setMsg(`saved ${r.saved} ratings for “${rater}” — agreement updated`);
      setSheet(null); setRater("");
      refresh(sel);
    } catch (e) { setErr(e.message); }
  };

  const items = sheet?.items || [];
  const levels = sheet?.rubric?.levels || [];

  return (
    <div>
      <div className="page-head">
        <h1 className="page-title">Raters</h1>
        <p className="page-sub">Score a study's answers as a human — on the same rubric the judge used —
          then measure how well the LLM judge agrees with humans (Krippendorff's α).</p>
      </div>

      {err && <div className="banner">{err}</div>}
      {msg && <div className="banner" style={{ borderLeftColor: "var(--green)" }}>{msg}</div>}

      <div className="card" style={{ marginBottom: 18 }}>
        <div className="field" style={{ marginBottom: 0 }}><label>Study</label>
          <select className="select" value={sel} onChange={(e) => pick(e.target.value)}>
            <option value="">— select a finished study —</option>
            {studies.map((s) => <option key={s.id} value={s.id}>{s.name}</option>)}
          </select></div>
      </div>

      {sel && (
        <>
          <div className="row-between" style={{ marginBottom: 12 }}>
            <div className="section-label" style={{ margin: 0 }}>Agreement</div>
            <div className="muted mono" style={{ fontSize: 12 }}>
              {raters.length} human rater{raters.length === 1 ? "" : "s"}
              {raters.length > 0 && ": "}{raters.map((r) => r.rater).join(" · ")}
            </div>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, alignItems: "stretch", marginBottom: 12 }}>
            <AlphaCard title="judge ↔ human" sub="does the LLM judge agree with the humans?" a={rel?.judge_vs_human} />
            <AlphaCard title="human ↔ human" sub="do the humans agree with each other?" a={rel?.human_ceiling} />
          </div>

          <div className="hint" style={{ marginBottom: 8, lineHeight: 1.7 }}>
            <b style={{ color: "var(--on-surface)" }}>Krippendorff's α</b> — inter-rater reliability
            ({rel?.metric || "—"} scale).{" "}
            <span style={{ color: "var(--green)" }}>■</span> ≥ 0.80 reliable ·{" "}
            <span style={{ color: "var(--amber-soft)" }}>■</span> 0.667–0.80 tentative ·{" "}
            <span style={{ color: "var(--error)" }}>■</span> &lt; 0.667 unreliable. Each α needs ≥ 2
            raters over the same answers.
          </div>
        </>
      )}

      {/* ── decoupled: add a new rating ─────────────────────────────────────── */}
      {sel && !sheet && (
        <div style={{ marginTop: 32, paddingTop: 24, borderTop: "1px solid var(--outline-variant)" }}>
          <div className="section-label" style={{ marginTop: 0 }}>Add a rating</div>
          <div className="card">
            <div className="field" style={{ marginBottom: 12 }}><label>Rater name</label>
              <input className="input mono" value={rater} onChange={(e) => setRater(e.target.value)}
                placeholder="e.g. alice" list="raters" />
              <datalist id="raters">{raters.map((r) => <option key={r.rater} value={r.rater} />)}</datalist>
              <div className="hint">pick an existing name to update those ratings, or enter a new one</div>
            </div>
            <button className="btn primary" disabled={!rater} onClick={start}>Start rating</button>
          </div>
        </div>
      )}

      {sheet && (
        <div style={{ marginTop: 32, paddingTop: 24, borderTop: "1px solid var(--outline-variant)" }}>
          <div className="section-label" style={{ marginTop: 0 }}>Rate as “{rater}” — {Object.keys(scores).length}/{items.length} scored</div>
          <RubricPanel rubric={sheet.rubric} judgeModel={sheet.judge_model} />
          {items.map((item) => (
            <div className="card" key={item.key} style={{ marginBottom: 10 }}>
              <div style={{ marginBottom: 6 }}><b>Q:</b> {item.question}</div>
              {item.reference && <div className="hint" style={{ marginBottom: 6 }}><b>Gold:</b> {item.reference}</div>}
              <div className="mono" style={{ fontSize: 13, marginBottom: 10, color: "var(--on-surface-variant)" }}>{item.answer}</div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                {levels.map((l) => (
                  <button key={l.value} title={l.description || l.label}
                    className={"btn btn-sm" + (scores[item.key] === l.value ? " primary" : "")}
                    onClick={() => setScores({ ...scores, [item.key]: l.value })}>
                    {l.value} <span className="muted" style={{ fontSize: 11 }}>{l.label}</span>
                  </button>
                ))}
              </div>
            </div>
          ))}
          <div className="row-between" style={{ marginTop: 12 }}>
            <button className="btn ghost" onClick={() => setSheet(null)}>cancel</button>
            <button className="btn primary" disabled={Object.keys(scores).length === 0} onClick={submit}>
              Submit {Object.keys(scores).length} ratings</button>
          </div>
        </div>
      )}
    </div>
  );
}
