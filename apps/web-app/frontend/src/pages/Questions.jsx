import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../lib/api.js";
import { useConfirm } from "../components/ConfirmModal.jsx";

// List of datasets. View a dataset or create a new one → separate pages.
export default function Questions() {
  const nav = useNavigate();
  const confirm = useConfirm();
  const [datasets, setDatasets] = useState(null);
  const [err, setErr] = useState(null);

  const load = () => api.datasets().then(setDatasets).catch((e) => setErr(e.message));
  useEffect(() => { load(); }, []);

  return (
    <div>
      <div className="page-head row-between">
        <div>
          <h1 className="page-title">Questions</h1>
          <p className="page-sub">Datasets your studies run on — questions with an optional gold
            reference for reference-guided judging.</p>
        </div>
        <button className="btn primary" onClick={() => nav("/questions/new")}>
          <span className="material-symbols-outlined" style={{ fontSize: 18, marginRight: 6 }}>add</span>New dataset</button>
      </div>

      {err && <div className="banner">{err}</div>}
      {!datasets && !err && <div className="loading">loading…</div>}
      {datasets && datasets.length === 0 && <div className="empty">no datasets yet — create one</div>}

      {datasets && datasets.length > 0 && (
        <div className="card table-wrap" style={{ padding: 0 }}>
          <table className="list-table">
            <thead><tr><th>Name</th><th>Questions</th><th>With reference</th><th></th></tr></thead>
            <tbody>
              {datasets.map((d) => (
                <tr key={d.id} style={{ cursor: "pointer" }} onClick={() => nav(`/questions/${d.id}`)}>
                  <td className="mono" style={{ color: "var(--amber-soft)" }}>{d.name}</td>
                  <td>{d.items.length}</td>
                  <td className="muted">{d.items.filter((i) => i.reference).length}</td>
                  <td style={{ textAlign: "right" }} onClick={(e) => e.stopPropagation()}>
                    <button className="btn btn-sm ghost" onClick={() => nav(`/questions/${d.id}`)}>view</button>{" "}
                    <button className="btn btn-sm danger"
                      onClick={() => confirm("Delete dataset?", `“${d.name}” will be removed.`,
                        () => api.deleteDataset(d.id).then(load).catch((e) => setErr(e.message)))}>delete</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
