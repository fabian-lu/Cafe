import { useEffect, useMemo, useState } from "react";
import { useParams } from "react-router-dom";
import { api } from "../lib/api.js";
import Breadcrumb from "../components/Breadcrumb.jsx";

const RESERVED = new Set(["id", "text", "question", "reference"]);

// A dataset on its own page — the questions + references as a table. Any extra item keys
// (topic, category, source, …) become filter dropdowns and extra columns automatically.
export default function ViewDataset() {
  const { id } = useParams();
  const [ds, setDs] = useState(null);
  const [err, setErr] = useState(null);
  const [fsel, setFsel] = useState({});   // metadata key -> selected value ("" = alle)

  useEffect(() => { api.dataset(id).then(setDs).catch((e) => setErr(e.message)); }, [id]);

  // filterable metadata keys: non-reserved, 2..30 distinct non-empty values
  const metaKeys = useMemo(() => {
    const vals = {};
    for (const it of ds?.items || []) {
      for (const [k, v] of Object.entries(it)) {
        if (RESERVED.has(k) || v == null || String(v).trim() === "") continue;
        (vals[k] ||= new Set()).add(String(v).trim());
      }
    }
    return Object.fromEntries(Object.entries(vals)
      .filter(([, v]) => v.size >= 2 && v.size <= 30)
      .map(([k, v]) => [k, [...v].sort()]));
  }, [ds]);

  const items = useMemo(() => (ds?.items || []).filter((it) =>
    Object.entries(fsel).every(([k, v]) => !v || String(it[k] ?? "").trim() === v)
  ), [ds, fsel]);

  const keys = Object.keys(metaKeys);

  return (
    <div>
      <Breadcrumb items={[{ label: "Questions", to: "/questions" }, { label: ds ? ds.name : "…" }]} />
      {err && <div className="banner">{err}</div>}
      {!ds && !err && <div className="loading">loading…</div>}
      {ds && (
        <>
          <div className="page-head">
            <h1 className="page-title">{ds.name}</h1>
            <p className="page-sub">{ds.items.length} questions · {ds.items.filter((i) => i.reference).length} with a reference answer.</p>
          </div>

          {keys.length > 0 && (
            <div style={{ display: "flex", gap: 12, marginBottom: 14, flexWrap: "wrap", alignItems: "center" }}>
              <span className="hint mono" style={{ margin: 0 }}>filter:</span>
              {keys.map((k) => (
                <label key={k} className="hint mono" style={{ margin: 0, display: "flex", alignItems: "center", gap: 6 }}>
                  {k}
                  <select className="select" style={{ width: "auto", padding: "4px 8px", fontSize: 12 }}
                    value={fsel[k] || ""} onChange={(e) => setFsel((s) => ({ ...s, [k]: e.target.value }))}>
                    <option value="">alle</option>
                    {metaKeys[k].map((v) => <option key={v} value={v}>{v}</option>)}
                  </select>
                </label>
              ))}
              {items.length !== ds.items.length && (
                <span className="hint" style={{ margin: 0 }}>→ {items.length} of {ds.items.length}</span>
              )}
            </div>
          )}

          <div className="card table-wrap" style={{ padding: 0 }}>
            <table className="list-table">
              <thead><tr>
                <th style={{ width: 44 }}>#</th><th>Question</th><th>Reference answer</th>
                {keys.map((k) => <th key={k}>{k}</th>)}
              </tr></thead>
              <tbody>
                {items.map((it, i) => (
                  <tr key={i}>
                    <td className="muted mono">{it.id ?? i + 1}</td>
                    <td>{it.text}</td>
                    <td className="muted">{it.reference || "—"}</td>
                    {keys.map((k) => <td key={k} className="muted mono" style={{ fontSize: 11 }}>{String(it[k] ?? "—")}</td>)}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
