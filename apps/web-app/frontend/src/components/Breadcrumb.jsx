import { Link } from "react-router-dom";

// DIVA-style breadcrumb: Parent › current. items = [{label, to?}, ...]; last has no `to`.
export default function Breadcrumb({ items }) {
  return (
    <div className="breadcrumb">
      {items.map((it, i) => (
        <span key={i} className="breadcrumb-item">
          {it.to ? <Link to={it.to}>{it.label}</Link> : <span className="current">{it.label}</span>}
          {i < items.length - 1 && <span className="material-symbols-outlined sep">chevron_right</span>}
        </span>
      ))}
    </div>
  );
}
