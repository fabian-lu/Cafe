import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../lib/api.js";
import NumberField from "../components/NumberField.jsx";
import JudgeModelField from "../components/JudgeModelField.jsx";

function parseLevels(str) {
  const parts = str.split(",").map((s) => s.trim()).filter(Boolean);
  if (parts.length && parts.every((p) => p !== "" && !Number.isNaN(Number(p)))) return parts.map(Number);
  return parts;
}

// One study row — clickable through to its overview page. Live (read-only) progress while running;
// run / re-run / archive / delete all live on the overview page, not here.
function StudyRow({ s, archived = false }) {
  const nav = useNavigate();
  const [prog, setProg] = useState(null);
  const es = useRef(null);

  useEffect(() => {
    if (s.status !== "running") return;
    const src = new EventSource(api.streamUrl(s.id));
    es.current = src;
    src.onmessage = (e) => { const p = JSON.parse(e.data); setProg(p); if (p.status === "done" || p.status === "failed") src.close(); };
    src.onerror = () => src.close();
    return () => src.close();
  }, [s.status, s.id]);

  const pct = prog && prog.total ? Math.round((prog.done / prog.total) * 100) : 0;
  const running = s.status === "running" || prog?.status === "running";
  const judging = prog?.phase === "judging";
  const go = () => nav(`/studies/${s.id}`);

  return (
    <tr className="clickable" onClick={go} style={archived ? { opacity: 0.72 } : undefined}>
      <td>
        <div className="mono">{s.name}</div>
        {s.description && <div className="muted" style={{ fontSize: 12, marginTop: 2, maxWidth: 360,
          overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{s.description}</div>}
      </td>
      <td className="muted mono" style={{ fontSize: 12 }}>{s.pipeline}</td>
      <td className="muted">{s.factors.map((f) => f.name).join(" · ") || "—"}</td>
      <td style={{ minWidth: 170 }}>
        {running ? (
          <div>
            <div className="progress"><div className={"progress-fill" + (judging ? " judging" : "")} style={{ width: `${pct}%` }} /></div>
            <div className="hint mono">{judging ? "judging" : "answering"} {prog?.total ? `${prog.done}/${prog.total}` : ""}</div>
          </div>
        ) : (
          <span className={"badge " + s.status}>{s.status}</span>
        )}
      </td>
      <td style={{ textAlign: "right", whiteSpace: "nowrap" }}>
        {s.status === "done" && (
          <button className="btn btn-sm" onClick={(e) => { e.stopPropagation(); nav(`/results?study=${s.id}`); }}>results</button>
        )}{" "}
        <span className="material-symbols-outlined row-arrow" style={{ fontSize: 18, verticalAlign: "-4px" }}>chevron_right</span>
      </td>
    </tr>
  );
}

function CreateStudy({ pipelines, datasets, rubrics, onCreated }) {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [pipeName, setPipeName] = useState(pipelines[0]?.name || "");
  const [datasetId, setDatasetId] = useState("");
  const [rubricId, setRubricId] = useState("");
  const [judge, setJudge] = useState("ollama_cloud/deepseek-v4-pro");
  const [reps, setReps] = useState(1);
  const [conc, setConc] = useState(8);
  const [stageSel, setStageSel] = useState({});
  const [paramVals, setParamVals] = useState({});
  const [err, setErr] = useState(null);

  const pipe = useMemo(() => pipelines.find((p) => p.name === pipeName) || pipelines[0], [pipelines, pipeName]);

  const stageParams = useMemo(() => {
    const out = {};
    for (const s of pipe?.stages || []) {
      const seen = new Map();
      for (const t of s.techniques) for (const p of t.params) seen.set(p.name, p.default);
      out[s.stage] = [...seen.entries()].map(([nm, def]) => ({ name: nm, default: def }));
    }
    return out;
  }, [pipe]);

  const toggle = (stage, tech) =>
    setStageSel((s) => ({ ...s, [stage]: { ...s[stage], [tech]: !s[stage]?.[tech] } }));

  const factors = useMemo(() => {
    const f = [];
    for (const s of pipe?.stages || []) {
      if (s.techniques.length > 1) {
        const levels = s.techniques.map((t) => t.name).filter((n) => stageSel[s.stage]?.[n]);
        if (levels.length) f.push({ name: s.stage, levels });
      }
      for (const p of stageParams[s.stage] || []) {
        const lv = parseLevels(paramVals[`${s.stage}.${p.name}`] || "");
        if (lv.length) f.push({ name: `${s.stage}.${p.name}`, levels: lv });
      }
    }
    return f;
  }, [pipe, stageSel, paramVals, stageParams]);

  const nConfigs = factors.reduce((acc, f) => acc * f.levels.length, factors.length ? 1 : 0);

  // A stage with >1 technique MUST have a factor (≥1 selected) or the pipeline can't pick one at run time.
  const unconfigured = useMemo(() =>
    (pipe?.stages || [])
      .filter((s) => s.techniques.length > 1 && !s.techniques.some((t) => stageSel[s.stage]?.[t.name]))
      .map((s) => s.stage), [pipe, stageSel]);

  const submit = async () => {
    setErr(null);
    try {
      const study = await api.createStudy({
        name, description, pipeline: pipeName, factors,
        dataset_id: datasetId ? Number(datasetId) : null,
        rubric_id: rubricId ? Number(rubricId) : null,
        judge_model: judge, replications: Number(reps), concurrency: Number(conc),
      });
      onCreated(study);
    } catch (e) { setErr(e.message); }
  };

  return (
    <div className="card">
      <div className="section-label" style={{ marginTop: 0 }}>New study</div>
      {err && <div className="banner">{err}</div>}

      <div className="row-between" style={{ gap: 12, flexWrap: "wrap" }}>
        <div className="field" style={{ flex: 2, minWidth: 200 }}><label>Name</label>
          <input className="input mono" value={name} onChange={(e) => setName(e.target.value)} placeholder="my-study" /></div>
        <div className="field" style={{ flex: 1, minWidth: 160 }}><label>Pipeline (system)</label>
          <select className="select" value={pipeName} onChange={(e) => { setPipeName(e.target.value); setStageSel({}); setParamVals({}); }}>
            {pipelines.map((p) => <option key={p.name} value={p.name}>{p.name}</option>)}
          </select></div>
      </div>
      <div className="field"><label>Description <span className="muted">— optional</span></label>
        <textarea className="input" rows={2} value={description} onChange={(e) => setDescription(e.target.value)}
          placeholder="what this study is testing, and why" /></div>

      <div className="section-label">Factors — vary the pipeline</div>
      {(pipe?.stages || []).map((s) => (
        <div key={s.stage} className="card" style={{ background: "var(--surface-container)", marginBottom: 12 }}>
          <div className="stage-name">{s.stage}</div>
          {s.techniques.length > 1 ? (
            <div style={{ marginBottom: stageParams[s.stage]?.length ? 12 : 0 }}>
              <div className="hint" style={{ marginBottom: 6 }}>techniques to compare (levels of this factor):</div>
              {s.techniques.map((t) => (
                <label key={t.name} className={"check" + (stageSel[s.stage]?.[t.name] ? " on" : "")}>
                  <input type="checkbox" checked={!!stageSel[s.stage]?.[t.name]} onChange={() => toggle(s.stage, t.name)} />
                  {t.name}
                </label>
              ))}
              {unconfigured.includes(s.stage) && (
                <div className="hint" style={{ color: "var(--error)", marginTop: 6 }}>
                  pick at least one — this stage has multiple techniques and needs a choice
                  (one = fixed, two or more = compared)</div>
              )}
            </div>
          ) : (
            <div className="hint" style={{ marginBottom: stageParams[s.stage]?.length ? 12 : 0 }}>
              single technique (<span className="mono">{s.techniques[0]?.name}</span>) — fixed stage, no factor needed</div>
          )}
          {(stageParams[s.stage] || []).map((p) => (
            <div key={p.name} className="field" style={{ marginBottom: 8 }}>
              <label>{s.stage}.{p.name} <span className="muted">— levels to sweep (comma-separated; default {String(p.default)})</span></label>
              <input className="input mono" value={paramVals[`${s.stage}.${p.name}`] || ""}
                onChange={(e) => setParamVals((v) => ({ ...v, [`${s.stage}.${p.name}`]: e.target.value }))}
                placeholder={String(p.default)} />
            </div>
          ))}
        </div>
      ))}

      <div className="row-between" style={{ gap: 12, flexWrap: "wrap" }}>
        <div className="field" style={{ flex: 1, minWidth: 180 }}><label>Dataset</label>
          <select className="select" value={datasetId} onChange={(e) => setDatasetId(e.target.value)}>
            <option value="">— select —</option>
            {datasets.map((d) => <option key={d.id} value={d.id}>{d.name} ({d.items.length})</option>)}
          </select></div>
        <div className="field" style={{ flex: 1, minWidth: 180 }}><label>Rubric</label>
          <select className="select" value={rubricId} onChange={(e) => setRubricId(e.target.value)}>
            <option value="">— select —</option>
            {rubrics.map((r) => <option key={r.id} value={r.id}>{r.name}</option>)}
          </select></div>
      </div>
      <div className="row-between" style={{ gap: 12, flexWrap: "wrap", alignItems: "flex-start" }}>
        <div className="field" style={{ flex: 1, minWidth: 220 }}><label>Judge model</label>
          <JudgeModelField value={judge} onChange={setJudge} /></div>
        <div className="field"><label>Replications <span className="muted">— per config</span></label>
          <NumberField value={reps} onChange={setReps} min={1} max={20} /></div>
        <div className="field"><label>Concurrency <span className="muted">— parallel calls</span></label>
          <NumberField value={conc} onChange={setConc} min={1} max={32} /></div>
      </div>

      <div className="row-between">
        <div className="hint">
          {unconfigured.length > 0
            ? <span style={{ color: "var(--error)" }}>configure {unconfigured.join(", ")} — each stage with multiple techniques needs a selection</span>
            : `${factors.length} factor${factors.length === 1 ? "" : "s"} · ${nConfigs} configuration${nConfigs === 1 ? "" : "s"}`}
        </div>
        <button className="btn primary" disabled={!name || factors.length === 0 || !datasetId || unconfigured.length > 0} onClick={submit}>Create study</button>
      </div>
    </div>
  );
}

export default function Studies() {
  const nav = useNavigate();
  const [pipelines, setPipelines] = useState(null);
  const [datasets, setDatasets] = useState([]);
  const [rubrics, setRubrics] = useState([]);
  const [studies, setStudies] = useState([]);
  const [archived, setArchived] = useState([]);
  const [showArchived, setShowArchived] = useState(false);
  const [creating, setCreating] = useState(false);
  const [err, setErr] = useState(null);

  const loadStudies = () => {
    api.studies().then(setStudies).catch((e) => setErr(e.message));
    api.studies(true).then(setArchived).catch(() => setArchived([]));
  };
  useEffect(() => {
    api.pipelines().then(setPipelines).catch((e) => setErr(e.message));
    api.datasets().then(setDatasets);
    api.rubrics().then(setRubrics);
    loadStudies();
  }, []);

  const canCreate = pipelines && pipelines.length > 0;

  return (
    <div>
      <div className="page-head row-between">
        <div>
          <h1 className="page-title">Studies</h1>
          <p className="page-sub">Define a factorial study over one of your pipelines, launch it, and
            explore the results. Each factor is a stage's techniques or a tunable parameter.</p>
        </div>
        {canCreate && <button className="btn primary" onClick={() => setCreating(!creating)}>
          {creating ? "Close" : "+ New study"}</button>}
      </div>

      {err && <div className="banner">{err}</div>}
      {pipelines && pipelines.length === 0 && <div className="banner">No pipelines discovered — add one in the Techniques folder.</div>}

      {creating && canCreate ? (
        <CreateStudy pipelines={pipelines} datasets={datasets} rubrics={rubrics}
          onCreated={(study) => nav(`/studies/${study.id}`)} />
      ) : (
        <>
          <div className="section-label">Studies</div>
          {studies.length === 0 ? <div className="empty">no studies yet — create one above</div> : (
            <div className="card table-wrap" style={{ padding: 0 }}>
              <table className="list-table">
                <thead><tr><th>Name</th><th>Pipeline</th><th>Factors</th><th>Status</th><th></th></tr></thead>
                <tbody>{studies.map((s) => <StudyRow key={s.id} s={s} />)}</tbody>
              </table>
            </div>
          )}

          {archived.length > 0 && (
            <>
              <button className="btn ghost btn-sm" style={{ marginTop: 20 }} onClick={() => setShowArchived((v) => !v)}>
                <span className="material-symbols-outlined" style={{ fontSize: 16, marginRight: 6, verticalAlign: "-3px" }}>
                  inventory_2</span>
                {showArchived ? "Hide" : "Show"} archived ({archived.length})
              </button>
              {showArchived && (
                <div className="card table-wrap" style={{ padding: 0, marginTop: 10 }}>
                  <table className="list-table">
                    <thead><tr><th>Name</th><th>Pipeline</th><th>Factors</th><th>Status</th><th></th></tr></thead>
                    <tbody>{archived.map((s) => <StudyRow key={s.id} s={s} archived />)}</tbody>
                  </table>
                </div>
              )}
            </>
          )}
        </>
      )}
    </div>
  );
}
