// A small amber ring spinner, optionally with a label. Factorial-Mono styling lives in index.css.
export default function Spinner({ label, size = 22 }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
      <span className="spinner" style={{ width: size, height: size, borderWidth: Math.max(2, Math.round(size / 9)) }} />
      {label && <span className="hint" style={{ margin: 0 }}>{label}</span>}
    </div>
  );
}
