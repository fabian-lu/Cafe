import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { api } from "../lib/api.js";
import InfoTip from "../components/InfoTip.jsx";
import Collapsible from "../components/Collapsible.jsx";

// ── formatting ────────────────────────────────────────────────────────────────
const fmt = (x, d = 3) => (x === null || x === undefined ? "—" : Number(x).toFixed(d));
const sig = (p) => (p == null ? "" : p < 0.001 ? "***" : p < 0.01 ? "**" : p < 0.05 ? "*" : p < 0.1 ? "." : "");
const pct = (x) => `${Math.round(x * 100)}%`;
const fmtSecs = (s) => {
  if (s == null) return "—";
  if (s < 60) return `${s < 10 ? s.toFixed(1) : Math.round(s)}s`;
  const m = Math.floor(s / 60), sec = Math.round(s % 60);
  if (m < 60) return `${m}m ${sec}s`;
  return `${Math.floor(m / 60)}h ${m % 60}m`;
};
const configLabel = (config) => Object.keys(config).sort().map((k) => `${k}=${config[k]}`).join(" · ");
const PALETTE = ["var(--amber)", "var(--green)", "var(--cyan)", "#c98bff", "#ff8b6b", "#7fd1ff"];

// ── glossary (hover explanations) ───────────────────────────────────────────────
const G = {
  F: <><b>F-statistic.</b> How much a factor moves quality relative to noise (between-group ÷ within-group variance). Bigger ⇒ a stronger, less-noisy effect.<br /><b>Example:</b> <span className="mono">F=130</span> ⇒ this factor's levels differ far more than chance would produce.</>,
  p: <><b>p-value.</b> The chance of seeing an effect this large if the factor truly did nothing. Below <span className="mono">0.05</span> ⇒ unlikely to be a fluke.<br /><b>Example:</b> <span className="mono">p=0.0001</span> ⇒ the effect is almost certainly real.</>,
  eta: <><b>Partial η².</b> Share of variance a factor explains, holding the others fixed. Rules of thumb: <span className="mono">0.01</span> small · <span className="mono">0.06</span> medium · <span className="mono">0.14</span> large.<br /><b>Example:</b> <span className="mono">η²=0.23</span> ⇒ this factor accounts for ~23% of the variance.</>,
  r2: <><b>R² (variance explained).</b> The share of <i>all</i> verdict variance the whole model captures. The rest is question difficulty + noise.<br /><b>Example:</b> <span className="mono">R²=0.27</span> ⇒ the factors jointly explain 27%.</>,
  d: <><b>Cohen's d.</b> Size of the gap between two levels, in standard deviations (scale-free). <span className="mono">0.2</span> small · <span className="mono">0.5</span> medium · <span className="mono">0.8</span> large. Sign shows which level scored higher.<br /><b>Example:</b> <span className="mono">d=+0.9</span> ⇒ a large improvement.</>,
  ci: <><b>95% confidence interval.</b> The plausible range for the true effect. If it doesn't cross 0, the effect is statistically significant.</>,
  beta: <><b>Log-odds (β).</b> The ordinal model's coefficient. <span className="mono">β&gt;0</span> ⇒ this level raises the odds of a <i>better</i> verdict; <span className="mono">β&lt;0</span> lowers them.<br /><b>Example:</b> <span className="mono">β=+1.15</span> shifts scores upward vs the baseline level.</>,
  or: <><b>Odds ratio (eᵝ).</b> How the odds of a higher verdict multiply for this level vs the baseline. <span className="mono">&gt;1</span> better, <span className="mono">&lt;1</span> worse.<br /><b>Example:</b> <span className="mono">3.2×</span> ⇒ about triple the odds of a better score.</>,
  z: <><b>z-statistic.</b> The coefficient divided by its standard error — how many SEs it sits from zero. <span className="mono">|z|&gt;1.96</span> ⇒ significant at 0.05.</>,
  pareto: <><b>Pareto-optimal.</b> A configuration that no other beats on <i>every</i> objective at once — you can't get more quality without paying more cost/latency/tokens.</>,
  vc: <><b>Variance components.</b> Splits the leftover (unexplained) variance into differences <i>between questions</i> vs plain <i>residual</i> noise. A large between-question share means some questions are just harder.</>,
  clmm: <><b>Cumulative-link mixed model.</b> Treats verdicts as <i>ordered categories</i> (0&lt;1&lt;2), not numbers — the statistically-correct model for an ordinal rubric. Includes a random effect per question.</>,
  marginal: <><b>Marginal mean.</b> The average quality at one factor level, averaged over all the other factors. "Is this level better on average?"</>,
};
const SigLegend = () => <span className="mono"> *** p&lt;.001 · ** &lt;.01 · * &lt;.05 · . &lt;.1</span>;

// ── small primitives ────────────────────────────────────────────────────────────
function Stat({ label, value, sub }) {
  return (
    <div style={{ background: "var(--surface-container)", borderRadius: 8, padding: "12px 14px" }}>
      <div className="hint mono" style={{ marginTop: 0 }}>{label}</div>
      <div style={{ fontFamily: "var(--font-display)", fontSize: 24, fontWeight: 700, color: "var(--amber-soft)", marginTop: 4, lineHeight: 1.1 }}>{value}</div>
      {sub && <div className="hint mono" style={{ marginTop: 4 }}>{sub}</div>}
    </div>
  );
}
const GRID = { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: 12 };
const AxisCap = ({ children }) => <div className="hint mono" style={{ textAlign: "center", marginTop: 6 }}>{children}</div>;

function Section({ title, hint, children }) {
  return (<><div className="section-label">{title}</div>
    {hint && <div className="hint" style={{ marginTop: -6, marginBottom: 10 }}>{hint}</div>}{children}</>);
}

// chart cursor tooltip
function useTip() {
  const [tip, setTip] = useState(null);
  return {
    tip,
    show: (e, content) => setTip({ x: e.clientX, y: e.clientY, content }),
    move: (e) => setTip((t) => (t ? { ...t, x: e.clientX, y: e.clientY } : t)),
    hide: () => setTip(null),
  };
}
const ChartTip = ({ tip }) => tip ? <div className="chart-tip" style={{ left: tip.x + 14, top: tip.y + 14 }}>{tip.content}</div> : null;

// horizontal value bar (shared by marginals / configs / latency)
function Bar({ label, value, valueText, n, max, color = "var(--amber)", labelColor, glow = true }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 8 }}>
      <div className="mono" title={label} style={{ width: 200, color: labelColor || "var(--on-surface-variant)", fontSize: 12,
        textAlign: "right", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{label}</div>
      <div style={{ flex: 1, background: "var(--surface-container-highest)", borderRadius: 6, height: 22, position: "relative" }}>
        <div style={{ width: pct(Math.min(1, value / max)), height: "100%", background: color, borderRadius: 6, boxShadow: glow ? "0 0 8px var(--amber-dim)" : "none" }} />
        <span className="mono" style={{ position: "absolute", right: 8, top: 3, fontSize: 12, color: "var(--on-surface)" }}>{valueText}</span>
      </div>
      {n != null && <div className="muted mono" style={{ width: 44, fontSize: 12 }}>n={n}</div>}
    </div>
  );
}

// ── attribution (F / p / η²) ─────────────────────────────────────────────────────
function EffectsTable({ effects, model, r2 }) {
  return (
    <div className="card" style={{ marginBottom: 16 }}>
      <table className="list-table">
        <thead><tr>
          <th>Factor</th>
          <th><InfoTip label="F">{G.F}</InfoTip></th>
          <th><InfoTip label="p">{G.p}</InfoTip></th>
          <th><InfoTip label="partial η²">{G.eta}</InfoTip></th>
          <th style={{ width: 200 }}>variance explained</th>
        </tr></thead>
        <tbody>
          {[...effects].sort((a, b) => (b.partial_eta_sq || 0) - (a.partial_eta_sq || 0)).map((e, i) => (
            <tr key={i}>
              <td className="mono">{e.factor}{e.interaction ? <span className="muted"> ×</span> : ""}</td>
              <td className="mono">{fmt(e.F, 2)}</td>
              <td className="mono">{fmt(e.p, 4)} <span style={{ color: "var(--amber)" }}>{sig(e.p)}</span></td>
              <td className="mono">{fmt(e.partial_eta_sq, 3)}</td>
              <td>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <div style={{ flex: 1, background: "var(--surface-container-highest)", height: 8, borderRadius: 4 }}>
                    <div style={{ width: pct(Math.min(1, e.partial_eta_sq || 0)), height: "100%", background: e.significant ? "var(--green)" : "var(--outline)", borderRadius: 4 }} />
                  </div>
                  <span className="mono muted" style={{ width: 42, fontSize: 11, textAlign: "right" }}>{pct(e.partial_eta_sq || 0)}</span>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="hint"><span style={{ color: "var(--green)" }}>■</span> significant at α=0.05 · <SigLegend />
        {r2 != null && <> · <InfoTip label="R²">{G.r2}</InfoTip> = <span className="mono" style={{ color: "var(--amber-soft)" }}>{pct(r2)}</span> of variance explained by the model</>}</div>
      {model && <div className="hint mono" style={{ marginTop: 2 }}>model: {model}</div>}
    </div>
  );
}

// ── marginal means / latency (per factor level) ──────────────────────────────────
function FactorBars({ groups, max, valueOf, valueText, cap }) {
  return groups.map((g) => (
    <div className="card" key={g.factor} style={{ marginBottom: 12 }}>
      <div className="stage-name">{g.factor}</div>
      {g.rows.map((r) => (
        <Bar key={r.level} label={String(r.level)} value={valueOf(r)} valueText={valueText(r)} n={r.n} max={max} />
      ))}
      {cap && <AxisCap>{cap}</AxisCap>}
    </div>
  ));
}

// ── interaction plot ─────────────────────────────────────────────────────────────
function InteractionChart({ records, factors, scaleMax }) {
  const [fa, setFa] = useState(factors[0]);
  const [fb, setFb] = useState(factors[1]);
  const t = useTip();
  const data = useMemo(() => {
    const aLevels = [...new Set(records.map((r) => String(r[fa])))].sort();
    const bLevels = [...new Set(records.map((r) => String(r[fb])))].sort();
    const cell = {};
    for (const r of records) { if (r.verdict == null) continue; (cell[`${r[fa]}|||${r[fb]}`] ||= []).push(r.verdict); }
    const mean = (a, b) => { const v = cell[`${a}|||${b}`]; return v && v.length ? v.reduce((s, x) => s + x, 0) / v.length : null; };
    const n = (a, b) => (cell[`${a}|||${b}`] || []).length;
    return { aLevels, bLevels, mean, n };
  }, [records, fa, fb]);

  const W = 540, H = 300, padL = 52, padR = 16, padT = 16, padB = 46, yMax = scaleMax || 2;
  const xPos = (i) => data.aLevels.length <= 1 ? padL + (W - padL - padR) / 2 : padL + (i * (W - padL - padR)) / (data.aLevels.length - 1);
  const yPos = (v) => H - padB - (v / yMax) * (H - padB - padT);

  return (
    <div className="card" style={{ marginBottom: 16 }}>
      <div className="row-between" style={{ marginBottom: 10, flexWrap: "wrap", gap: 8 }}>
        <div className="stage-name" style={{ margin: 0 }}>how <span className="mono" style={{ color: "var(--amber-soft)" }}>{fa}</span> depends on <span className="mono" style={{ color: "var(--amber-soft)" }}>{fb}</span></div>
        <div style={{ display: "flex", gap: 8 }}>
          <select className="select" style={{ width: "auto", padding: "4px 8px", fontSize: 12 }} value={fa} onChange={(e) => setFa(e.target.value)}>
            {factors.map((f) => <option key={f} value={f} disabled={f === fb}>{f}</option>)}</select>
          <select className="select" style={{ width: "auto", padding: "4px 8px", fontSize: 12 }} value={fb} onChange={(e) => setFb(e.target.value)}>
            {factors.map((f) => <option key={f} value={f} disabled={f === fa}>{f}</option>)}</select>
        </div>
      </div>
      <svg width={W} height={H} style={{ maxWidth: "100%" }}>
        {[0, 0.25, 0.5, 0.75, 1].map((f) => {
          const y = yPos(yMax * f);
          return <g key={f}><line x1={padL} y1={y} x2={W - padR} y2={y} stroke="var(--outline-variant)" strokeDasharray="2 4" />
            <text x={padL - 8} y={y + 4} textAnchor="end" fill="var(--outline)" fontSize="10" fontFamily="var(--font-mono)">{(yMax * f).toFixed(1)}</text></g>;
        })}
        {data.bLevels.map((b, bi) => {
          const pts = data.aLevels.map((a, ai) => ({ a, ai, m: data.mean(a, b) })).filter((p) => p.m != null);
          const color = PALETTE[bi % PALETTE.length];
          return <g key={b}>
            <polyline fill="none" stroke={color} strokeWidth="2" points={pts.map((p) => `${xPos(p.ai)},${yPos(p.m)}`).join(" ")} />
            {pts.map((p) => <circle key={p.a} cx={xPos(p.ai)} cy={yPos(p.m)} r="5" fill={color} stroke="#000" strokeWidth="0.5"
              style={{ cursor: "pointer" }}
              onMouseEnter={(e) => t.show(e, <>{fa}={p.a} · {fb}={b}<br />mean {fmt(p.m, 2)} <span className="muted">(n={data.n(p.a, b)})</span></>)}
              onMouseMove={t.move} onMouseLeave={t.hide} />)}
          </g>;
        })}
        {data.aLevels.map((a, ai) => <text key={a} x={xPos(ai)} y={H - padB + 16} textAnchor="middle" fill="var(--on-surface-variant)" fontSize="11" fontFamily="var(--font-mono)">{a}</text>)}
        <text x={W / 2} y={H - 6} textAnchor="middle" fill="var(--outline)" fontSize="11" fontFamily="var(--font-mono)">{fa} →</text>
        <text x={13} y={(H - padB + padT) / 2} textAnchor="middle" fill="var(--outline)" fontSize="11" fontFamily="var(--font-mono)" transform={`rotate(-90 13 ${(H - padB + padT) / 2})`}>mean quality →</text>
      </svg>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 12, marginTop: 4 }}>
        {data.bLevels.map((b, bi) => <div key={b} className="mono" style={{ fontSize: 12, display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{ width: 12, height: 3, background: PALETTE[bi % PALETTE.length], display: "inline-block", borderRadius: 2 }} />{fb}={b}</div>)}
      </div>
      <div className="hint" style={{ marginTop: 8 }}>parallel lines ⇒ no interaction; lines that cross or fan ⇒ the effect of one factor depends on the other.</div>
      <ChartTip tip={t.tip} />
    </div>
  );
}

// ── configuration ranking ────────────────────────────────────────────────────────
function ConfigRanking({ configMeans, best, scaleMax }) {
  const rows = [...configMeans].sort((a, b) => b.mean - a.mean);
  const max = scaleMax || Math.max(...rows.map((r) => r.mean), 0.0001);
  const bestLabel = best ? configLabel(best.config) : null;
  return (
    <div className="card" style={{ marginBottom: 16 }}>
      {rows.map((r) => {
        const label = configLabel(r.config);
        const isBest = label === bestLabel;
        return <Bar key={label} label={label} value={r.mean} valueText={fmt(r.mean, 2)} n={r.n} max={max}
          color={isBest ? "var(--green)" : "var(--amber-dim)"} labelColor={isBest ? "var(--green)" : undefined} glow={false} />;
      })}
      <AxisCap>→ mean quality (0–{max})</AxisCap>
      <div className="hint"><span style={{ color: "var(--green)" }}>■</span> best configuration</div>
    </div>
  );
}

// ── Cohen's d forest ─────────────────────────────────────────────────────────────
function Forest({ pairwise }) {
  const items = pairwise.filter((d) => d.d != null);
  if (!items.length) return null;
  const M = Math.max(...items.map((d) => Math.max(Math.abs(d.ci_low ?? d.d), Math.abs(d.ci_high ?? d.d), Math.abs(d.d))), 0.2) * 1.1;
  const scale = (v) => ((v + M) / (2 * M)) * 100;
  return (
    <div className="card" style={{ marginBottom: 16 }}>
      {items.map((d, i) => {
        const label = `${d.factor}: ${d.level_a} vs ${d.level_b}`;
        const excludesZero = d.ci_low != null && d.ci_high != null && (d.ci_low > 0) === (d.ci_high > 0);
        const color = excludesZero ? "var(--green)" : "var(--outline)";
        return (
          <div key={i} style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 10 }}>
            <div className="mono" title={label} style={{ width: 220, fontSize: 12, textAlign: "right", color: "var(--on-surface-variant)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{label}</div>
            <div style={{ flex: 1, position: "relative", height: 22 }}>
              <div style={{ position: "absolute", left: "50%", top: 0, bottom: 0, width: 1, background: "var(--outline-variant)" }} />
              {d.ci_low != null && <div style={{ position: "absolute", top: 10, height: 2, background: color, left: `${scale(d.ci_low)}%`, width: `${scale(d.ci_high) - scale(d.ci_low)}%` }} />}
              <div style={{ position: "absolute", top: 6, width: 10, height: 10, borderRadius: "50%", background: color, left: `calc(${scale(d.d)}% - 5px)` }} />
            </div>
            <div className="mono" style={{ width: 132, fontSize: 12, color }}>{d.d >= 0 ? "+" : ""}{fmt(d.d, 2)}
              {d.ci_low != null && <span className="muted"> [{fmt(d.ci_low, 2)},{fmt(d.ci_high, 2)}]</span>}</div>
          </div>
        );
      })}
      {/* axis */}
      <div style={{ display: "flex", gap: 12, marginTop: 2 }}>
        <div style={{ width: 220 }} />
        <div style={{ flex: 1, position: "relative", height: 14 }} className="mono muted">
          <span style={{ position: "absolute", left: 0, fontSize: 10 }}>−{M.toFixed(1)}</span>
          <span style={{ position: "absolute", left: "50%", transform: "translateX(-50%)", fontSize: 10 }}>0</span>
          <span style={{ position: "absolute", right: 0, fontSize: 10 }}>+{M.toFixed(1)}</span>
        </div>
        <div style={{ width: 132 }} />
      </div>
      <div className="hint" style={{ marginTop: 4 }}><InfoTip label="Cohen's d">{G.d}</InfoTip> ± 95% <InfoTip label="CI">{G.ci}</InfoTip> ·
        vertical line = 0 (no difference) · <span style={{ color: "var(--green)" }}>green</span> = CI excludes 0.</div>
    </div>
  );
}

// ── ordinal CLMM ─────────────────────────────────────────────────────────────────
function CLMMTable({ clmm }) {
  if (!clmm.available) return <div className="card" style={{ marginBottom: 16 }}>
    <div className="hint">ordinal model not fitted{clmm.reason ? ` — ${clmm.reason}` : ""}. Install R + the <span className="mono">ordinal</span> package (<span className="mono">cafe doctor</span>) to enable it.</div></div>;
  const coefs = [...clmm.coefficients].sort((a, b) => Math.abs(b.estimate || 0) - Math.abs(a.estimate || 0));
  return (
    <div className="card" style={{ marginBottom: 16 }}>
      <table className="list-table">
        <thead><tr><th>Term</th>
          <th><InfoTip label="log-odds (β)">{G.beta}</InfoTip></th>
          <th><InfoTip label="odds ratio">{G.or}</InfoTip></th>
          <th><InfoTip label="z">{G.z}</InfoTip></th>
          <th><InfoTip label="p">{G.p}</InfoTip></th></tr></thead>
        <tbody>
          {coefs.map((c, i) => (
            <tr key={i}>
              <td className="mono" title={c.label || c.term} style={{ color: c.significant ? "var(--green)" : undefined, maxWidth: 320, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{c.label || c.term}{c.interaction ? <span className="muted"> ×</span> : ""}</td>
              <td className="mono">{c.estimate >= 0 ? "+" : ""}{fmt(c.estimate, 3)}</td>
              <td className="mono">{fmt(Math.exp(c.estimate), 2)}×</td>
              <td className="mono">{fmt(c.z, 2)}</td>
              <td className="mono">{fmt(c.p, 4)} <span style={{ color: "var(--amber)" }}>{sig(c.p)}</span></td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="hint"><span style={{ color: "var(--green)" }}>■</span> significant at α=0.05 · <SigLegend /> · β &gt; 0 ⇒ higher odds of a better verdict · {clmm.n_obs} obs{clmm.log_lik != null ? ` · logLik ${fmt(clmm.log_lik, 1)}` : ""}</div>
    </div>
  );
}

// ── binary logistic ──────────────────────────────────────────────────────────────
function LogisticTable({ logistic }) {
  if (!logistic.available) return <div className="card" style={{ marginBottom: 16 }}>
    <div className="hint">logistic model not fitted{logistic.reason ? ` — ${logistic.reason}` : ""}.</div></div>;
  const terms = [...logistic.terms].sort((a, b) => Math.abs(b.coef || 0) - Math.abs(a.coef || 0));
  return (
    <div className="card" style={{ marginBottom: 16 }}>
      <table className="list-table">
        <thead><tr><th>Term</th>
          <th><InfoTip label="log-odds (β)">{G.beta}</InfoTip></th>
          <th><InfoTip label="odds ratio">{G.or}</InfoTip></th>
          <th><InfoTip label="p">{G.p}</InfoTip></th></tr></thead>
        <tbody>
          {terms.map((t, i) => (
            <tr key={i}>
              <td className="mono" title={t.label} style={{ color: t.significant ? "var(--green)" : undefined }}>{t.label}{t.interaction ? <span className="muted"> ×</span> : ""}</td>
              <td className="mono">{t.coef >= 0 ? "+" : ""}{fmt(t.coef, 3)}</td>
              <td className="mono">{fmt(t.odds_ratio, 2)}×</td>
              <td className="mono">{fmt(t.p, 4)} <span style={{ color: "var(--amber)" }}>{sig(t.p)}</span></td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="hint"><span style={{ color: "var(--green)" }}>■</span> significant at α=0.05 · <SigLegend /> · {logistic.n_obs} obs</div>
    </div>
  );
}

// ── vertical verdict distribution ────────────────────────────────────────────────
function Distribution({ records, rubric }) {
  const counts = useMemo(() => {
    const c = {}; for (const r of records) if (r.verdict != null) c[r.verdict] = (c[r.verdict] || 0) + 1; return c;
  }, [records]);
  const levels = rubric?.levels?.length ? rubric.levels
    : [...new Set(records.map((r) => r.verdict).filter((v) => v != null))].sort((a, b) => a - b).map((v) => ({ value: v, label: "" }));
  const total = Object.values(counts).reduce((a, b) => a + b, 0) || 1;
  const max = Math.max(...levels.map((l) => counts[l.value] || 0), 1);
  const H = 200;
  return (
    <div className="card" style={{ marginBottom: 16 }}>
      <div style={{ display: "flex", gap: 12 }}>
        <div className="mono muted" style={{ fontSize: 11, writingMode: "vertical-rl", transform: "rotate(180deg)", textAlign: "center" }}>count →</div>
        <div style={{ flex: 1 }}>
          <div style={{ display: "flex", alignItems: "flex-end", gap: 28, height: H, borderBottom: "1px solid var(--outline-variant)" }}>
            {levels.map((l) => {
              const n = counts[l.value] || 0;
              return (
                <div key={l.value} style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "flex-end", height: "100%" }}>
                  <div className="mono" style={{ fontSize: 12, marginBottom: 6 }}>{n} <span className="muted">· {pct(n / total)}</span></div>
                  <div style={{ width: 72, maxWidth: "72%", height: `${(n / max) * 100}%`, minHeight: n ? 3 : 0, background: "var(--amber)", borderRadius: "6px 6px 0 0", boxShadow: "0 0 8px var(--amber-dim)" }} />
                </div>
              );
            })}
          </div>
          <div style={{ display: "flex", gap: 28, marginTop: 8 }}>
            {levels.map((l) => <div key={l.value} className="mono" style={{ flex: 1, textAlign: "center", fontSize: 12 }}>
              <span style={{ color: "var(--amber-soft)" }}>{l.value}</span>{l.label ? ` ${l.label}` : ""}</div>)}
          </div>
          <AxisCap>verdict level →</AxisCap>
        </div>
      </div>
      <div className="hint" style={{ marginTop: 6 }}>how the {total} verdicts are spread across the rubric levels.</div>
    </div>
  );
}

// ── variance components ──────────────────────────────────────────────────────────
function VarComponents({ vc }) {
  const tot = (vc.random_intercept || 0) + (vc.residual || 0) || 1;
  const seg = [["between-question", vc.random_intercept || 0, "var(--cyan)"], ["residual", vc.residual || 0, "var(--outline)"]];
  return (
    <div className="card" style={{ marginBottom: 16 }}>
      <div style={{ display: "flex", height: 26, borderRadius: 6, overflow: "hidden", marginBottom: 10 }}>
        {seg.map(([lab, v, c]) => <div key={lab} style={{ width: pct(v / tot), background: c }} title={`${lab}: ${fmt(v, 3)}`} />)}
      </div>
      {seg.map(([lab, v, c]) => <div key={lab} className="mono" style={{ fontSize: 12, display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
        <span style={{ width: 12, height: 12, background: c, borderRadius: 3, display: "inline-block" }} />{lab}: σ² = {fmt(v, 3)} <span className="muted">({pct(v / tot)})</span></div>)}
    </div>
  );
}

// ── cost / quality Pareto ────────────────────────────────────────────────────────
function Pareto({ pareto }) {
  const objectives = (pareto?.objectives || []).filter((o) => o !== "quality");
  const [xKey, setXKey] = useState(objectives[0]);
  const t = useTip();
  if (!pareto || !objectives.length || !pareto.rows?.length) return null;
  const W = 540, H = 320, pad = 52;
  const xs = pareto.rows.map((r) => r[xKey]), ys = pareto.rows.map((r) => r.quality);
  const xmin = Math.min(...xs), xmax = Math.max(...xs), ymin = Math.min(...ys), ymax = Math.max(...ys);
  const sx = (v) => pad + ((v - xmin) / (xmax - xmin || 1)) * (W - 2 * pad);
  const sy = (v) => H - pad - ((v - ymin) / (ymax - ymin || 1)) * (H - 2 * pad);
  const frontier = pareto.rows.filter((r) => r.pareto_optimal).sort((a, b) => a[xKey] - b[xKey]);
  return (
    <div className="card" style={{ marginBottom: 16 }}>
      <div className="row-between" style={{ marginBottom: 8, flexWrap: "wrap", gap: 8 }}>
        <div className="stage-name" style={{ margin: 0 }}>quality vs {xKey}</div>
        {objectives.length > 1 && <div style={{ display: "flex", gap: 6 }}>
          {objectives.map((o) => <button key={o} className={"btn btn-sm" + (o === xKey ? " primary" : "")} onClick={() => setXKey(o)}>{o}</button>)}</div>}
      </div>
      <svg width={W} height={H} style={{ maxWidth: "100%" }}>
        <line x1={pad} y1={H - pad} x2={W - pad} y2={H - pad} stroke="var(--outline-variant)" />
        <line x1={pad} y1={pad} x2={pad} y2={H - pad} stroke="var(--outline-variant)" />
        {[0, 0.5, 1].map((f) => { const yv = ymin + f * (ymax - ymin); const y = sy(yv);
          return <text key={f} x={pad - 8} y={y + 4} textAnchor="end" fill="var(--outline)" fontSize="10" fontFamily="var(--font-mono)">{yv.toFixed(1)}</text>; })}
        {[0, 0.5, 1].map((f) => { const xv = xmin + f * (xmax - xmin); const x = sx(xv);
          return <text key={f} x={x} y={H - pad + 14} textAnchor="middle" fill="var(--outline)" fontSize="10" fontFamily="var(--font-mono)">{xv < 0.01 ? xv.toExponential(1) : xv.toFixed(2)}</text>; })}
        <polyline fill="none" stroke="var(--amber-dim)" strokeWidth="1.5" strokeDasharray="4 3" points={frontier.map((r) => `${sx(r[xKey])},${sy(r.quality)}`).join(" ")} />
        {pareto.rows.map((r, i) => {
          const opt = r.pareto_optimal;
          return <circle key={i} cx={sx(r[xKey])} cy={sy(r.quality)} r={opt ? 6 : 4} fill={opt ? "var(--amber)" : "var(--outline)"} stroke="#000" strokeWidth={opt ? 1 : 0}
            style={{ cursor: "pointer" }}
            onMouseEnter={(e) => t.show(e, <>{opt ? "★ " : ""}{configLabel(r.config)}<br />quality {fmt(r.quality, 2)} · {xKey} {fmt(r[xKey], 4)}</>)}
            onMouseMove={t.move} onMouseLeave={t.hide} />;
        })}
        <text x={W / 2} y={H - 8} textAnchor="middle" fill="var(--outline)" fontSize="11" fontFamily="var(--font-mono)">{xKey} → (lower is better)</text>
        <text x={14} y={H / 2} textAnchor="middle" fill="var(--outline)" fontSize="11" fontFamily="var(--font-mono)" transform={`rotate(-90 14 ${H / 2})`}>quality →</text>
      </svg>
      <div className="hint" style={{ marginBottom: 8 }}>★ amber = <InfoTip label="Pareto-optimal">{G.pareto}</InfoTip> · {frontier.length} of {pareto.rows.length} configs · hover a dot for its config</div>
      <div className="mono" style={{ fontSize: 12 }}>
        <div className="hint mono" style={{ marginBottom: 4 }}>the frontier — your best trade-offs:</div>
        {frontier.map((r, i) => <div key={i} title={configLabel(r.config)} style={{ color: "var(--amber-soft)", marginBottom: 2, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
          ★ {configLabel(r.config)} <span className="muted">— quality {fmt(r.quality, 2)} · {xKey} {fmt(r[xKey], 4)}</span></div>)}
      </div>
      <ChartTip tip={t.tip} />
    </div>
  );
}

// ── answers (filterable) ─────────────────────────────────────────────────────────
function Records({ records, factors }) {
  const [vf, setVf] = useState("");
  const verdicts = useMemo(() => [...new Set(records.map((r) => r.verdict).filter((v) => v != null))].sort((a, b) => a - b), [records]);
  const filtered = vf === "" ? records : records.filter((r) => String(r.verdict) === vf);
  const shown = filtered.slice(0, 80);
  return (
    <div className="card">
      <div className="row-between" style={{ marginBottom: 10, flexWrap: "wrap", gap: 8 }}>
        <div className="hint mono" style={{ margin: 0 }}>{records.length} answers</div>
        <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
          <span className="hint mono" style={{ margin: 0 }}>verdict:</span>
          <button className={"btn btn-sm" + (vf === "" ? " primary" : "")} onClick={() => setVf("")}>all</button>
          {verdicts.map((v) => <button key={v} className={"btn btn-sm" + (String(v) === vf ? " primary" : "")} onClick={() => setVf(String(v))}>{v}</button>)}
        </div>
      </div>
      <table className="list-table">
        <thead><tr><th>Question</th><th>Config</th><th>Answer</th><th>Verdict</th></tr></thead>
        <tbody>
          {shown.map((r, i) => (
            <tr key={i}>
              <td title={r.question || r.input_id} style={{ maxWidth: 220 }}>{(r.question || r.input_id || "").slice(0, 110)}</td>
              <td className="muted mono" style={{ fontSize: 11 }} title={factors.map((f) => `${f}=${r[f]}`).join(" ")}>{factors.map((f) => `${f}=${r[f]}`).join(" ")}</td>
              <td style={{ maxWidth: 300 }}>
                <div title={r.answer || ""}>{(r.answer || "").slice(0, 160)}</div>
                {r.reasoning && <div className="hint" title={r.reasoning} style={{ marginTop: 4, fontStyle: "italic" }}>{r.reasoning.slice(0, 160)}</div>}
              </td>
              <td className="mono" style={{ color: "var(--amber-soft)", fontSize: 15 }}>{r.verdict ?? "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {shown.length < filtered.length && <div className="hint">showing {shown.length} of {filtered.length}</div>}
    </div>
  );
}

// ── page ──────────────────────────────────────────────────────────────────────
export default function Results() {
  const [params, setParams] = useSearchParams();
  const [studies, setStudies] = useState([]);
  const [sel, setSel] = useState(params.get("study") || "");
  const [res, setRes] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => { api.studies().then((all) => {
    const done = all.filter((s) => s.status === "done");
    setStudies(done);
    if (!sel && done.length) setSel(String(done[0].id));
  }); }, []);

  useEffect(() => {
    if (!sel) return;
    setErr(null); setRes(null);
    setParams({ study: sel });
    api.results(sel).then(setRes).catch((e) => setErr(e.message));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sel]);

  const byFactor = useMemo(() => {
    const g = {};
    (res?.marginal_means || []).forEach((r) => { (g[r.factor] ||= []).push(r); });
    return Object.entries(g).map(([factor, rows]) => ({ factor, rows }));
  }, [res]);

  const latency = useMemo(() => {
    const recs = res?.records || [], facs = res?.factors || [];
    return facs.map((f) => {
      const byLevel = {};
      for (const r of recs) { if (r.elapsed_s == null) continue; (byLevel[r[f]] ||= []).push(r.elapsed_s); }
      const rows = Object.entries(byLevel).map(([level, arr]) => ({ level, mean: arr.reduce((a, b) => a + b, 0) / arr.length, n: arr.length })).sort((a, b) => a.level.localeCompare(b.level));
      return { factor: f, rows };
    }).filter((g) => g.rows.length);
  }, [res]);

  const scaleMax = useMemo(() => {
    const vals = (res?.rubric?.levels || []).map((l) => l.value);
    return vals.length ? Math.max(...vals) : null;
  }, [res]);

  const factors = res?.factors || [];
  const scale = res?.rubric?.scale_type;
  const marginalMax = scaleMax || Math.max(...(res?.marginal_means || []).map((r) => r.mean), 0.0001);
  const latMax = Math.max(...latency.flatMap((g) => g.rows.map((r) => r.mean)), 0.0001);

  return (
    <div>
      <div className="page-head row-between">
        <div>
          <h1 className="page-title">Results</h1>
          <p className="page-sub">Attribution, effect sizes, the scale-correct model, and the cost/quality
            frontier — every number computed by cafe-core.</p>
        </div>
        <select className="select" style={{ width: 240 }} value={sel} onChange={(e) => setSel(e.target.value)}>
          <option value="">— select a study —</option>
          {studies.map((s) => <option key={s.id} value={s.id}>{s.name}</option>)}
        </select>
      </div>

      {err && <div className="banner">{err}</div>}
      {!sel && <div className="empty">select a finished study above</div>}
      {sel && !res && !err && <div className="loading">loading results…</div>}

      {res && (
        <>
          {/* headline */}
          <div className="card" style={{ marginBottom: 16 }}>
            <div style={GRID}>
              <Stat label="overall mean quality" value={fmt(res.overall_mean, 2)} sub={scaleMax != null ? `of ${scaleMax}` : null} />
              <Stat label="answers" value={res.timing?.n_answers ?? res.records?.length ?? "—"} />
              <Stat label="configurations" value={res.config_means?.length ?? "—"} />
              <Stat label={<InfoTip label="variance explained">{G.r2}</InfoTip>} value={res.r_squared != null ? pct(res.r_squared) : "—"} />
            </div>
            {res.best_config && (
              <div style={{ marginTop: 14, paddingTop: 14, borderTop: "1px solid var(--outline-variant)" }}>
                <span className="hint mono">best configuration</span>
                <div className="mono" title={configLabel(res.best_config.config)} style={{ color: "var(--green)", marginTop: 4, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {configLabel(res.best_config.config)} <span className="muted">— mean {fmt(res.best_config.mean, 2)}</span></div>
              </div>
            )}
          </div>

          {res.effects?.length > 0 && (
            <Section title="Attribution — which factor drives quality"
              hint="linear mixed-effects model (verdict ~ factors + random question). The correct model for a numeric scale, and the general effect-size view for any scale; the scale-matched model is below.">
              <EffectsTable effects={res.effects} model={res.effects_model} r2={res.r_squared} />
            </Section>
          )}

          {byFactor.length > 0 && (
            <Section title="Marginal means — is a level better on average" hint="mean quality at each factor level, averaged over the others.">
              <FactorBars groups={byFactor} max={marginalMax} valueOf={(r) => r.mean} valueText={(r) => fmt(r.mean, 2)} cap={`→ mean quality (0–${marginalMax})`} />
            </Section>
          )}

          {factors.length >= 2 && res.records?.length > 0 && (
            <Section title="Interaction — do two factors combine">
              <InteractionChart records={res.records} factors={factors} scaleMax={scaleMax} />
            </Section>
          )}

          {res.config_means?.length > 1 && (
            <Section title="Configurations — the full ranking" hint="mean quality of every configuration, best highlighted.">
              <ConfigRanking configMeans={res.config_means} best={res.best_config} scaleMax={scaleMax} />
            </Section>
          )}

          {res.pairwise_d?.length > 0 && (
            <Section title="Effect sizes — how big is the gap">
              <Forest pairwise={res.pairwise_d} />
            </Section>
          )}

          {scale === "ordinal" && res.clmm && (
            <Section title="Ordinal model — cumulative-link mixed model"
              hint="the statistically-correct model for your ordinal rubric: verdicts as ordered categories, not numbers.">
              <CLMMTable clmm={res.clmm} />
            </Section>
          )}
          {scale === "binary" && res.logistic && (
            <Section title="Binary model — logistic regression"
              hint="the statistically-correct model for your binary (pass/fail) rubric: log-odds of a pass.">
              <LogisticTable logistic={res.logistic} />
            </Section>
          )}
          {scale === "numeric" && (
            <div className="hint" style={{ marginBottom: 16 }}>For your numeric rubric, the linear mixed-effects model above is the statistically-correct model.</div>
          )}

          {res.variance_components && (
            <Section title="Variance components" hint={<>how the leftover variance splits — see <InfoTip label="variance components">{G.vc}</InfoTip>.</>}>
              <VarComponents vc={res.variance_components} />
            </Section>
          )}

          {latency.length > 0 && (
            <Section title="Latency by factor level" hint="mean wall-time per answer at each level — which levels are slow.">
              <FactorBars groups={latency} max={latMax} valueOf={(r) => r.mean} valueText={(r) => fmtSecs(r.mean)} cap="→ mean latency per answer" />
            </Section>
          )}

          {res.records?.length > 0 && (
            <Section title="Verdict distribution">
              <Distribution records={res.records} rubric={res.rubric} />
            </Section>
          )}

          {res.timing && (
            <Section title="Run timing">
              <Timing timing={res.timing} />
            </Section>
          )}

          {res.pareto && (
            <Section title="Cost / quality" hint="which configurations give the most quality per unit cost, latency, or tokens.">
              <Pareto pareto={res.pareto} />
            </Section>
          )}

          {res.records?.length > 0 && (
            <Collapsible title="Answers — question → answer → verdict" hint={`${res.records.length} rows`}>
              <Records records={res.records} factors={factors} />
            </Collapsible>
          )}

          {res.report && (
            <Collapsible title="Full text report" hint="copy-paste for a paper">
              <pre style={{ background: "var(--surface-container-lowest)", border: "1px solid var(--outline-variant)", borderRadius: 8, padding: 16, fontFamily: "var(--font-mono)", fontSize: 12, whiteSpace: "pre-wrap", color: "var(--on-surface-variant)", overflow: "auto" }}>{res.report}</pre>
            </Collapsible>
          )}
        </>
      )}
    </div>
  );
}

// ── run timing (kept at bottom to reuse fmtSecs) ────────────────────────────────
function Timing({ timing }) {
  const entries = Object.entries(timing.per_stage_s || {}).sort((a, b) => b[1] - a[1]);
  const total = entries.reduce((a, [, v]) => a + v, 0) || 1;
  const max = Math.max(...entries.map(([, v]) => v), 0.0001);
  return (
    <div className="card" style={{ marginBottom: 16 }}>
      <div style={{ ...GRID, marginBottom: entries.length ? 18 : 0 }}>
        <Stat label="total time" value={fmtSecs(timing.run_wall_s ?? timing.wall_s)} />
        <Stat label="total compute" value={fmtSecs(timing.total_compute_s)} />
        <Stat label="answers" value={timing.n_answers} />
        <Stat label="mean / answer" value={fmtSecs(timing.mean_cell_s)} />
      </div>
      {entries.length > 0 && <>
        <div className="hint mono" style={{ marginBottom: 8 }}>time by stage</div>
        {entries.map(([stage, secs]) => (
          <div key={stage} style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 8 }}>
            <div className="mono" style={{ width: 120, textAlign: "right", fontSize: 13, color: "var(--on-surface-variant)" }}>{stage}</div>
            <div style={{ flex: 1, background: "var(--surface-container-highest)", borderRadius: 6, height: 20 }}>
              <div style={{ width: pct(secs / max), height: "100%", background: "var(--amber)", borderRadius: 6, boxShadow: "0 0 8px var(--amber-dim)" }} />
            </div>
            <div className="mono muted" style={{ width: 96, fontSize: 12 }}>{fmtSecs(secs)} · {pct(secs / total)}</div>
          </div>
        ))}
      </>}
      <div className="hint" style={{ marginTop: 10 }}>total time is the run's wall clock; the per-stage split is answer-generation time from cafe's traces.</div>
    </div>
  );
}
