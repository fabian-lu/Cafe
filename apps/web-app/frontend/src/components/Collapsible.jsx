import { useState } from "react";

// A section that stays collapsed until the user opens it. `hint` shows next to the toggle.
export default function Collapsible({ title, hint, defaultOpen = false, children }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div style={{ marginBottom: 16 }}>
      <button className="btn ghost" style={{ width: "100%", justifyContent: "space-between", display: "flex" }}
        onClick={() => setOpen((v) => !v)}>
        <span><span className="material-symbols-outlined" style={{ fontSize: 18, marginRight: 8, verticalAlign: "-4px" }}>
          {open ? "expand_more" : "chevron_right"}</span>{title}</span>
        {hint && <span className="muted mono" style={{ fontSize: 12 }}>{hint}</span>}
      </button>
      {open && <div style={{ marginTop: 12 }}>{children}</div>}
    </div>
  );
}
