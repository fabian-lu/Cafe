import { useState } from "react";

// A friendly judge-model picker. The generate models are short aliases the *pipeline* resolves
// to full provider ids; the judge is passed straight to LiteLLM, so it needs the provider-qualified
// id (e.g. "ollama_cloud/deepseek-v4-pro"). This dropdown hides that: friendly labels, full ids under
// the hood, plus a "Custom…" escape hatch for any other LiteLLM model string.
const PRESETS = [
  { id: "ollama_cloud/deepseek-v4-pro", label: "deepseek-v4-pro — recommended judge (different family)" },
  { id: "ollama_cloud/gpt-oss-120b", label: "gpt-oss-120b" },
  { id: "ollama_cloud/gpt-oss-20b", label: "gpt-oss-20b" },
  { id: "ollama_cloud/gemma3:4b", label: "gemma3-4b" },
];

export default function JudgeModelField({ value, onChange }) {
  const isPreset = PRESETS.some((p) => p.id === value);
  const [custom, setCustom] = useState(!isPreset && !!value);

  return (
    <div>
      <select className="select" value={custom ? "__custom__" : value}
        onChange={(e) => {
          if (e.target.value === "__custom__") { setCustom(true); onChange(""); }
          else { setCustom(false); onChange(e.target.value); }
        }}>
        {PRESETS.map((p) => <option key={p.id} value={p.id}>{p.label}</option>)}
        <option value="__custom__">Custom model…</option>
      </select>
      {custom && (
        <input className="input mono" style={{ marginTop: 8 }} value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder="provider/model — e.g. openai/gpt-4o-mini" />
      )}
    </div>
  );
}
