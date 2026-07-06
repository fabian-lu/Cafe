import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../lib/api.js";
import { useConfirm } from "../components/ConfirmModal.jsx";

// Full defaults per scale type (levels WITH descriptions).
const DEFAULTS = {
  ordinal: [
    { value: 0, label: "incorrect", description: "Factually incorrect or off-topic." },
    { value: 1, label: "partial", description: "Partially correct." },
    { value: 2, label: "correct", description: "Fully correct; wording does not matter." },
  ],
  binary: [
    { value: 0, label: "fail", description: "Incorrect, misleading, or unsupported." },
    { value: 1, label: "pass", description: "Correct and adequately supported." },
  ],
  numeric: [
    { value: 0, label: "useless", description: "No help — wrong, empty, or off-topic." },
    { value: 5, label: "partial", description: "Somewhat helpful, with notable gaps." },
    { value: 10, label: "ideal", description: "Complete, correct, and maximally helpful." },
  ],
};
const DEFAULT_INSTRUCTION =
  "Grade whether the ANSWER correctly answers the QUESTION, using the REFERENCE as the gold answer. " +
  "Judge substance, not style or length; award partial credit for a partially-correct answer.";
const DEFAULT_SYSTEM = "You are a strict, fair, impartial evaluator.";

export default function Rubrics() {
  const confirm = useConfirm();
  const nav = useNavigate();
  const [rubrics, setRubrics] = useState([]);
  const [presets, setPresets] = useState([]);
  const [err, setErr] = useState(null);

  // form
  const [name, setName] = useState("");
  const [scale, setScale] = useState("ordinal");
  const [levels, setLevels] = useState(DEFAULTS.ordinal);
  const [instruction, setInstruction] = useState(DEFAULT_INSTRUCTION);
  const [preset, setPreset] = useState("reference_qa");
  const [useCustom, setUseCustom] = useState(false);
  const [tmpl, setTmpl] = useState("");
  const [system, setSystem] = useState(DEFAULT_SYSTEM);
  // preview example
  const [exQ, setExQ] = useState("What is the capital of France?");
  const [exA, setExA] = useState("Paris.");
  const [exR, setExR] = useState("Paris is the capital of France.");
  const [preview, setPreview] = useState(null);

  const load = () => api.rubrics().then(setRubrics).catch((e) => setErr(e.message));
  useEffect(() => { load(); api.judgePresets().then(setPresets).catch(() => {}); }, []);

  const changeScale = (v) => { setScale(v); setLevels(DEFAULTS[v]); };
  const setLevel = (i, k, val) =>
    setLevels(levels.map((l, j) => (j === i ? { ...l, [k]: k === "value" ? Number(val) : val } : l)));
  const isBinary = scale === "binary";

  const rubricPayload = () => ({
    name: name || "rubric", scale_type: scale, levels, instruction,
    preset, system_prompt: system || null,
    prompt_template: useCustom && tmpl ? tmpl : null,
  });

  const runPreview = async () => {
    setErr(null); setPreview(null);
    try {
      const r = await api.judgePreview({
        rubric: rubricPayload(), judge_model: "preview", system_prompt: system || null,
        question: exQ, answer: exA, reference: exR || null,
      });
      setPreview(r.preview);
    } catch (e) { setErr(e.message); }
  };

  const submit = async () => {
    setErr(null);
    try {
      await api.createRubric(rubricPayload());
      setName(""); setPreview(null); load();
    } catch (e) { setErr(e.message); }
  };

  const field = (label, hint, node) => (
    <div className="field">
      <label>{label}{hint && <span className="muted"> — {hint}</span>}</label>
      {node}
    </div>
  );

  return (
    <div>
      <div className="page-head">
        <h1 className="page-title">Rubrics</h1>
        <p className="page-sub">The full grading spec: the scale, the levels, and how the judge is
          prompted. Scale type drives the statistics:
          <span className="mono"> ordinal → CLMM</span>, numeric → linear, binary → logistic.</p>
      </div>

      {err && <div className="banner">{err}</div>}

      <div className="card">
        <div className="section-label" style={{ marginTop: 0 }}>New rubric</div>

        <div className="row-between" style={{ gap: 12, flexWrap: "wrap" }}>
          {field("Name", null,
            <input className="input mono" value={name} onChange={(e) => setName(e.target.value)} placeholder="correctness_0_2" />)}
          <div className="field" style={{ width: 200 }}>
            <label>Scale type</label>
            <select className="select" value={scale} onChange={(e) => changeScale(e.target.value)}>
              <option value="ordinal">ordinal (CLMM)</option>
              <option value="numeric">numeric (linear)</option>
              <option value="binary">binary (logistic)</option>
            </select>
          </div>
        </div>

        {field("Levels", isBinary ? "binary — fixed 0/1" : scale === "numeric" ? "anchors; the judge may return any integer in range" : "value · label · description",
          <div>
            {levels.map((l, i) => (
              <div key={i} className="row-between" style={{ marginBottom: 8 }}>
                <input className="input mono" style={{ width: 66 }} type="number" value={l.value}
                  disabled={isBinary} onChange={(e) => setLevel(i, "value", e.target.value)} />
                <input className="input mono" style={{ width: 140 }} value={l.label} placeholder="label"
                  onChange={(e) => setLevel(i, "label", e.target.value)} />
                <input className="input" value={l.description} placeholder="description"
                  onChange={(e) => setLevel(i, "description", e.target.value)} />
                {!isBinary && <button className="btn btn-sm danger" onClick={() => setLevels(levels.filter((_, j) => j !== i))}>✕</button>}
              </div>
            ))}
            {!isBinary && <button className="btn btn-sm ghost" onClick={() => setLevels([...levels, { value: levels.length, label: "", description: "" }])}>+ level</button>}
          </div>)}

        {field("Instruction", "the task told to the judge",
          <textarea className="input" rows={2} value={instruction} onChange={(e) => setInstruction(e.target.value)} />)}

        {field("System prompt", "the judge's system message",
          <textarea className="input" rows={2} value={system} onChange={(e) => setSystem(e.target.value)} />)}

        {field("Judge prompt", "choose a preset CAFE ships, or write your own template",
          <div>
            <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap", marginBottom: 8 }}>
              <select className="select" style={{ width: 220 }} value={preset} disabled={useCustom}
                onChange={(e) => setPreset(e.target.value)}>
                {presets.map((p) => <option key={p.name} value={p.name}>{p.name}</option>)}
              </select>
              <label className={"check" + (useCustom ? " on" : "")}>
                <input type="checkbox" checked={useCustom} onChange={(e) => setUseCustom(e.target.checked)} />
                custom template
              </label>
            </div>
            {useCustom && (
              <textarea className="input mono" style={{ fontSize: 12 }} rows={7} value={tmpl}
                onChange={(e) => setTmpl(e.target.value)}
                placeholder={"{instruction}\n\nQuestion: {question}\nReference: {reference}\nAnswer: {answer}\n\nScore:\n{scale}\n\nGRADE: <{grade}>"} />
            )}
          </div>)}

        <div className="section-label">Preview the judge prompt</div>
        <div className="card" style={{ background: "var(--surface-container)" }}>
          <div className="row-between" style={{ gap: 12, flexWrap: "wrap" }}>
            {field("Example question", null, <input className="input" value={exQ} onChange={(e) => setExQ(e.target.value)} />)}
          </div>
          <div className="row-between" style={{ gap: 12, flexWrap: "wrap" }}>
            <div className="field" style={{ flex: 1, minWidth: 200 }}><label>Example answer</label>
              <input className="input" value={exA} onChange={(e) => setExA(e.target.value)} /></div>
            <div className="field" style={{ flex: 1, minWidth: 200 }}><label>Example reference</label>
              <input className="input" value={exR} onChange={(e) => setExR(e.target.value)} /></div>
          </div>
          <button className="btn" onClick={runPreview}>Preview judge prompt</button>
          {preview && (
            <pre style={{ marginTop: 14, background: "var(--surface-container-lowest)", border: "1px solid var(--outline-variant)",
              borderRadius: 8, padding: 16, fontFamily: "var(--font-mono)", fontSize: 12, whiteSpace: "pre-wrap",
              color: "var(--on-surface-variant)", maxHeight: 320, overflow: "auto" }}>{preview}</pre>
          )}
        </div>

        <div style={{ marginTop: 18, textAlign: "right" }}>
          <button className="btn primary" disabled={!name || levels.length < 2} onClick={submit}>Create rubric</button>
        </div>
      </div>

      <div className="section-label">Existing rubrics</div>
      {rubrics.length === 0 ? <div className="empty">no rubrics yet</div> : (
        <div className="card table-wrap" style={{ padding: 0 }}>
          <table className="list-table">
            <thead><tr><th>Name</th><th>Scale</th><th>Levels</th><th>Prompt</th><th></th></tr></thead>
            <tbody>
              {rubrics.map((r) => (
                <tr key={r.id}>
                  <td className="mono">{r.name}</td>
                  <td>{r.scale_type}</td>
                  <td className="muted">{r.levels.map((l) => l.value).join(" · ")}</td>
                  <td className="muted mono" style={{ fontSize: 12 }}>{r.prompt_template ? "custom" : (r.preset || "reference_qa")}</td>
                  <td style={{ textAlign: "right", whiteSpace: "nowrap" }}>
                    <button className="btn btn-sm ghost" onClick={() => nav(`/rubrics/${r.id}`)}>view</button>{" "}
                    <button className="btn btn-sm danger"
                      onClick={() => confirm("Delete rubric?", `“${r.name}” will be removed.`,
                        () => api.deleteRubric(r.id).then(load).catch((e) => setErr(e.message)))}>delete</button>
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
