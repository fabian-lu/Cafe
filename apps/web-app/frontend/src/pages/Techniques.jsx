import { useEffect, useState } from "react";
import { api } from "../lib/api.js";

// The systems under test, discovered from techniques/ (one per .py with a @pipe.compose).
// Pick one from the dropdown to inspect its stages/techniques. Reload re-scans the folder.
export default function Techniques() {
  const [pipes, setPipes] = useState(null);
  const [sel, setSel] = useState("");
  const [error, setError] = useState(null);
  const [reloading, setReloading] = useState(false);

  const apply = (list) => {
    setPipes(list);
    setSel((cur) => (list.some((p) => p.name === cur) ? cur : list[0]?.name || ""));
  };
  const load = (reload) => {
    setError(null);
    (reload ? api.reloadPipelines() : api.pipelines())
      .then(apply).catch((e) => setError(e.message)).finally(() => setReloading(false));
  };
  useEffect(() => { load(false); }, []);
  const reload = () => { setReloading(true); load(true); };

  const pipe = pipes?.find((p) => p.name === sel);

  return (
    <div>
      <div className="page-head row-between">
        <div>
          <h1 className="page-title">Techniques</h1>
          <p className="page-sub">The systems under test, discovered from your{" "}
            <span className="mono">techniques/</span> folder — one per <span className="mono">.py</span>{" "}
            file with a <span className="mono">@pipe.compose</span>. Select a pipeline to inspect it.</p>
        </div>
        <button className="btn" onClick={reload} disabled={reloading}>
          <span className="material-symbols-outlined" style={{ fontSize: 18, marginRight: 6 }}>
            {reloading ? "hourglass_empty" : "refresh"}</span>
          {reloading ? "scanning…" : "Reload"}
        </button>
      </div>

      {error && <div className="banner">Could not load pipelines: {error}</div>}
      {!pipes && !error && <div className="loading">loading pipelines…</div>}
      {pipes && pipes.length === 0 && (
        <div className="empty">no pipelines — add a <span className="mono">.py</span> with a
          <span className="mono"> @pipe.compose</span> to techniques/ and hit Reload</div>
      )}

      {pipes && pipes.length > 0 && (
        <>
          <div className="card" style={{ marginBottom: 16 }}>
            <div className="field" style={{ marginBottom: 0 }}>
              <label>Pipeline ({pipes.length} discovered)</label>
              <select className="select" value={sel} onChange={(e) => setSel(e.target.value)}>
                {pipes.map((p) => <option key={p.name} value={p.name}>{p.name}</option>)}
              </select>
            </div>
          </div>

          {pipe && (
            <div className="card">
              <div className="row-between" style={{ marginBottom: 16 }}>
                <div className="mono" style={{ fontSize: 15, color: "var(--amber-soft)" }}>
                  <span className="material-symbols-outlined" style={{ fontSize: 18, marginRight: 8, color: "var(--amber)" }}>account_tree</span>
                  {pipe.name}</div>
                <div className="muted mono" style={{ fontSize: 12 }}>{pipe.stages.map((s) => s.stage).join(" → ")}</div>
              </div>
              {pipe.stages.map((s) => (
                <div key={s.stage} style={{ marginBottom: 14 }}>
                  <div className="stage-name">{s.stage}</div>
                  <div className="tech-grid">
                    {s.techniques.map((t) => (
                      <div className="tech" key={t.name}>
                        <div className="tech-name">{t.name}</div>
                        {t.description && <div className="tech-meta">{t.description}</div>}
                        <div>
                          {t.params.map((p) => <span className="chip param" key={p.name}>{p.name}={String(p.default)}</span>)}
                          {t.cost_usd > 0 && <span className="chip">${t.cost_usd}/call</span>}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
