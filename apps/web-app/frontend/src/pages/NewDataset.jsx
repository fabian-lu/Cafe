import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../lib/api.js";
import Breadcrumb from "../components/Breadcrumb.jsx";

const BLANK = { text: "", reference: "" };

// DIVA-style entry: separate fields per question (question + optional reference answer), add/remove rows.
export default function NewDataset() {
  const nav = useNavigate();
  const [name, setName] = useState("");
  const [rows, setRows] = useState([{ ...BLANK }, { ...BLANK }]);
  const [err, setErr] = useState(null);

  const set = (i, k, v) => setRows(rows.map((r, j) => (j === i ? { ...r, [k]: v } : r)));
  const add = () => setRows([...rows, { ...BLANK }]);
  const remove = (i) => setRows(rows.filter((_, j) => j !== i));

  const valid = rows.filter((r) => r.text.trim());
  const submit = async () => {
    setErr(null);
    try {
      const items = valid.map((r, i) => ({ id: `q${i}`, text: r.text.trim(), reference: r.reference.trim() }));
      const ds = await api.createDataset({ name, items });
      nav(`/questions/${ds.id}`);
    } catch (e) { setErr(e.message); }
  };

  return (
    <div>
      <Breadcrumb items={[{ label: "Questions", to: "/questions" }, { label: "New dataset" }]} />
      <div className="page-head">
        <h1 className="page-title">New dataset</h1>
        <p className="page-sub">A set of questions your studies run on. The reference answer is optional
          (used for reference-guided judging).</p>
      </div>

      {err && <div className="banner">{err}</div>}

      <div className="card">
        <div className="field"><label>Dataset name</label>
          <input className="input mono" value={name} onChange={(e) => setName(e.target.value)}
            placeholder="e.g. misconceptions-10" /></div>

        <div className="section-label">Questions</div>
        {rows.map((r, i) => (
          <div className="card" key={i} style={{ background: "var(--surface-container)", marginBottom: 10 }}>
            <div className="row-between" style={{ marginBottom: 8 }}>
              <span className="mono muted" style={{ fontSize: 12 }}>Question {i + 1}</span>
              {rows.length > 1 && <button className="btn btn-sm danger" onClick={() => remove(i)}>remove</button>}
            </div>
            <div className="field" style={{ marginBottom: 10 }}><label>Question</label>
              <textarea className="input" rows={2} value={r.text} onChange={(e) => set(i, "text", e.target.value)}
                placeholder="What is the capital of France?" /></div>
            <div className="field" style={{ marginBottom: 0 }}><label>Reference answer <span className="muted">(optional)</span></label>
              <input className="input" value={r.reference} onChange={(e) => set(i, "reference", e.target.value)}
                placeholder="Paris" /></div>
          </div>
        ))}
        <button className="btn btn-sm ghost" onClick={add}>+ add question</button>

        <div className="row-between" style={{ marginTop: 18 }}>
          <div className="hint">{valid.length} question{valid.length === 1 ? "" : "s"} · {valid.filter((r) => r.reference.trim()).length} with a reference</div>
          <div>
            <button className="btn ghost" onClick={() => nav("/questions")}>Cancel</button>{" "}
            <button className="btn primary" disabled={!name || valid.length === 0} onClick={submit}>Create dataset</button>
          </div>
        </div>
      </div>
    </div>
  );
}
