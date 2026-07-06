import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { api } from "../lib/api.js";
import Breadcrumb from "../components/Breadcrumb.jsx";

// A dataset on its own page — the questions + references as a table.
export default function ViewDataset() {
  const { id } = useParams();
  const [ds, setDs] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => { api.dataset(id).then(setDs).catch((e) => setErr(e.message)); }, [id]);

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
          <div className="card table-wrap" style={{ padding: 0 }}>
            <table className="list-table">
              <thead><tr><th style={{ width: 44 }}>#</th><th>Question</th><th>Reference answer</th></tr></thead>
              <tbody>
                {ds.items.map((it, i) => (
                  <tr key={i}>
                    <td className="muted mono">{i + 1}</td>
                    <td>{it.text}</td>
                    <td className="muted">{it.reference || "—"}</td>
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
