import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { api } from "../lib/api.js";
import Breadcrumb from "../components/Breadcrumb.jsx";

// A rubric on its own page — its scale, levels, judge prompt config, and a live preview.
export default function ViewRubric() {
  const { id } = useParams();
  const [r, setR] = useState(null);
  const [preview, setPreview] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => { api.rubric(id).then(setR).catch((e) => setErr(e.message)); }, [id]);

  useEffect(() => {
    if (!r) return;
    api.judgePreview({
      rubric: r, judge_model: "preview", system_prompt: r.system_prompt || null,
      question: "What is the capital of France?", answer: "Paris.",
      reference: "Paris is the capital of France.",
    }).then((x) => setPreview(x.preview)).catch(() => setPreview(null));
  }, [r]);

  const row = (k, v) => (
    <div style={{ display: "flex", gap: 14, marginBottom: 10 }}>
      <div className="mono muted" style={{ width: 130, flexShrink: 0, fontSize: 12, textAlign: "right", paddingTop: 2 }}>{k}</div>
      <div style={{ flex: 1 }}>{v}</div>
    </div>
  );

  return (
    <div>
      <Breadcrumb items={[{ label: "Rubrics", to: "/rubrics" }, { label: r ? r.name : "…" }]} />
      {err && <div className="banner">{err}</div>}
      {!r && !err && <div className="loading">loading…</div>}
      {r && (
        <>
          <div className="page-head">
            <h1 className="page-title">{r.name}</h1>
            <p className="page-sub"><span className="mono">{r.scale_type}</span> scale ·{" "}
              {r.prompt_template ? "custom prompt template" : `“${r.preset}” preset`}</p>
          </div>

          <div className="card" style={{ marginBottom: 16 }}>
            {row("scale type", <span className="mono">{r.scale_type}</span>)}
            {row("levels",
              <div>{r.levels.map((l) => (
                <div key={l.value} style={{ marginBottom: 4 }}>
                  <span className="mono" style={{ color: "var(--amber-soft)" }}>{l.value}</span>{" "}
                  <b>{l.label}</b> <span className="muted">— {l.description || "—"}</span>
                </div>
              ))}</div>)}
            {row("instruction", r.instruction || <span className="muted">—</span>)}
            {row("system prompt", r.system_prompt || <span className="muted">(default)</span>)}
            {row("judge prompt", r.prompt_template
              ? <span className="muted">custom template</span>
              : <span className="mono">{r.preset}</span>)}
            {r.prompt_template && row("template",
              <pre className="mono" style={{ margin: 0, whiteSpace: "pre-wrap", fontSize: 12, color: "var(--on-surface-variant)" }}>{r.prompt_template}</pre>)}
          </div>

          <div className="section-label">Judge prompt preview</div>
          {preview
            ? <pre style={{ background: "var(--surface-container-lowest)", border: "1px solid var(--outline-variant)",
                borderRadius: 8, padding: 16, fontFamily: "var(--font-mono)", fontSize: 12, whiteSpace: "pre-wrap",
                color: "var(--on-surface-variant)", maxHeight: 400, overflow: "auto" }}>{preview}</pre>
            : <div className="loading">rendering…</div>}
        </>
      )}
    </div>
  );
}
