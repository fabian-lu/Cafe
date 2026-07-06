import { NavLink } from "react-router-dom";
import MugIcon from "./MugIcon.jsx";

const NAV = [
  { to: "/studies", label: "Studies", icon: "science" },
  { to: "/questions", label: "Questions", icon: "quiz" },
  { to: "/rubrics", label: "Rubrics", icon: "rule" },
  { to: "/techniques", label: "Techniques", icon: "widgets" },
  { to: "/results", label: "Results", icon: "insights" },
  { to: "/raters", label: "Raters", icon: "groups" },
];

export default function Sidebar() {
  return (
    <nav className="sidebar">
      <div className="brand">
        <MugIcon />
        <div>
          <div className="brand-word">CAFE<span className="dot">.</span></div>
          <div className="brand-tag">Factorial Eval</div>
        </div>
      </div>

      {NAV.map((item) => (
        <NavLink key={item.to} to={item.to}
          className={({ isActive }) => "nav-link" + (isActive ? " active" : "")}>
          <span className="material-symbols-outlined">{item.icon}</span>
          <span>{item.label}</span>
        </NavLink>
      ))}

      <div className="sidebar-foot">v0.1.0</div>
    </nav>
  );
}
