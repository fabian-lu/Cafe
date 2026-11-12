import { useEffect, useMemo, useState } from "react";
import { useParams } from "react-router-dom";
import { api, answerText } from "../lib/api.js";
import Breadcrumb from "../components/Breadcrumb.jsx";

const RESERVED = new Set(["id", "text", "question", "reference"]);

// An item's value for a key: an item may have any keys ("constructor" included), so only its own.
const valueOf = (it, k) => (Object.prototype.hasOwnProperty.call(it, k) ? it[k] : undefined);
// A value as table text: an item may hold any JSON value (a number, a dict, a list), shown as JSON.
const isEmpty = (v) => v == null || v === "";
const cell = (v) => (isEmpty(v) ? "—" : answerText(v));

// A dataset on its own page — the questions + references as a table. Any extra item keys
// (topic, category, source, …) become filter dropdowns and extra columns automatically.
export default function ViewDataset() {
  const { id } = useParams();
  const [ds, setDs] = useState(null);
  const [err, setErr] = useState(null);
  const [fsel, setFsel] = useState(new Map());   // metadata key -> selected value ("" = all)

  useEffect(() => { api.dataset(id).then(setDs).catch((e) => setErr(e.message)); }, [id]);

  // filterable metadata keys: non-reserved, 2..30 distinct non-empty values (Maps, not objects:
  // a key named like an Object member, e.g. "constructor", must not resolve to that member)
  const metaKeys = useMemo(() => {
    const vals = new Map();
    for (const it of ds?.items || []) {
      for (const [k, v] of Object.entries(it)) {
        if (RESERVED.has(k) || v == null || answerText(v).trim() === "") continue;
        if (!vals.has(k)) vals.set(k, new Set());
        vals.get(k).add(answerText(v).trim());
      }
    }
    return new Map([...vals]
      .filter(([, v]) => v.size >= 2 && v.size <= 30)
      .map(([k, v]) => [k, [...v].sort()]));
  }, [ds]);

  const items = useMemo(() => (ds?.items || []).filter((it) =>
    [...fsel].every(([k, v]) => !v || answerText(valueOf(it, k)).trim() === v)
  ), [ds, fsel]);

  const keys = [...metaKeys.keys()];

  return (
    <div>
      <Breadcrumb items={[{ label: "Questions", to: "/questions" }, { label: ds ? ds.name : "…" }]} />
      {err && <div className="banner">{err}</div>}
      {!ds && !err && <div className="loading">loading…</div>}
      {ds && (
        <>
          <div className="page-head">
            <h1 className="page-title">{ds.name}</h1>
            <p className="page-sub">{ds.items.length} questions · {ds.items.filter((i) => !isEmpty(i.reference)).length} with a reference answer.</p>
          </div>

          {keys.length > 0 && (
            <div style={{ display: "flex", gap: 12, marginBottom: 14, flexWrap: "wrap", alignItems: "center" }}>
              <span className="hint mono" style={{ margin: 0 }}>filter:</span>
              {keys.map((k) => (
                <label key={k} className="hint mono" style={{ margin: 0, display: "flex", alignItems: "center", gap: 6 }}>
                  {k}
                  <select className="select" style={{ width: "auto", padding: "4px 8px", fontSize: 12 }}
                    value={fsel.get(k) || ""} onChange={(e) => setFsel((s) => new Map(s).set(k, e.target.value))}>
                    <option value="">all</option>
                    {metaKeys.get(k).map((v) => <option key={v} value={v}>{v}</option>)}
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
                    <td className="muted mono">{it.id == null ? i + 1 : answerText(it.id)}</td>
                    <td>{answerText(it.text)}</td>
                    <td className="muted">{cell(it.reference)}</td>
                    {keys.map((k) => <td key={k} className="muted mono" style={{ fontSize: 11 }}>{cell(valueOf(it, k))}</td>)}
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
