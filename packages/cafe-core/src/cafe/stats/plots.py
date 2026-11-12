"""Plots for an evaluation — a small, good-looking menu.

    result.plot()                 # dashboard of the key plots
    result.plot("marginals")      # a single plot
    ax = result.plot("interaction", factors=["model", "prompt"])  # returns the Axes

Each function returns the matplotlib ``Axes`` (the dashboard returns the ``Figure``),
so you can `.savefig(...)` or tweak it. Needs the ``stats`` extra (matplotlib + pandas).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cafe.evaluation import Evaluation

# Factorial-Mono-ish palette: amber accent + muted companions.
ACCENT = "#f5a623"
PALETTE = ["#f5a623", "#3b7dd8", "#46b29d", "#c0504d", "#8e6fb5", "#d99a2b"]
_GREY = "#9aa0a6"


# ── helpers ──────────────────────────────────────────────────────────────────────

def _short(x: Any, maxlen: int = 16) -> str:
    """Shorten a long level label (e.g. a model string) for an axis tick."""
    return _cut(str(x).split("/")[-1], maxlen)


def _cut(s: str, maxlen: int) -> str:
    """``s`` cut to ``maxlen`` characters, ending in an ellipsis if it was longer."""
    return s if len(s) <= maxlen else s[: maxlen - 1] + "…"


def _labels(levels: list[Any], maxlen: int = 16) -> list[str]:
    """Labels for one factor's levels: their :func:`_short` forms, kept distinct.

    Levels whose short forms collide (``openai/gpt-4o-mini`` and ``azure/gpt-4o-mini``, or
    two prompts that share their first ``maxlen`` characters) show more of their text, the
    whole string cut at ``2 * maxlen``; if even that doesn't tell them apart, they're
    numbered (``… (1)``, ``… (2)``). Levels that don't collide keep their short form.
    """
    labels = [_short(x, maxlen) for x in levels]
    for short in dict.fromkeys(labels):
        same = [i for i, lab in enumerate(labels) if lab == short]
        if len(same) < 2:
            continue
        longer = [_cut(str(levels[i]), 2 * maxlen) for i in same]
        if len(set(longer)) < len(same):
            longer = [f"{short} ({n})" for n in range(1, len(same) + 1)]
        for i, lab in zip(same, longer):
            labels[i] = lab
    return labels


def _cfg_labels(keys: list[Any]) -> list[str]:
    """Compact config labels — level *values* only (the factor order is in the title),
    e.g. ``keyword·2·none`` instead of ``retrieve=keyword·retrieve.top_k=2·rerank=none``.
    Each factor's levels are labelled with :func:`_labels`, so distinct configs read
    differently."""
    rows = [k if isinstance(k, tuple) else (k,) for k in keys]
    columns = []
    for col in zip(*rows):
        levels = list(dict.fromkeys(col))
        columns.append(dict(zip(levels, _labels(levels))))
    return ["·".join(columns[j][v] for j, v in enumerate(row)) for row in rows]


def _frame(evaluation: "Evaluation"):
    from cafe.stats._frame import analysis_frame

    if getattr(evaluation, "ratings", None) is None:
        raise ValueError("plotting needs a judged evaluation (ratings) — run evaluate() with a judge")
    df = analysis_frame(evaluation.ratings)
    if df.empty or "verdict" not in df.columns:
        raise ValueError("no usable verdicts to plot")
    # Only factors that actually vary — a pinned single-level factor (e.g. generate=grounded)
    # adds nothing to a plot and just clutters the labels.
    factors = [f for f in evaluation.ratings.factors if f in df.columns and df[f].nunique() > 1]
    if not factors:
        raise ValueError("no factor varies across ≥2 levels — nothing to plot")
    return df, factors


def _rubric_range(evaluation: "Evaluation") -> tuple[float, float]:
    r = getattr(evaluation.ratings, "rubric", None)
    if r is not None:
        return float(r.min_value), float(r.max_value)
    return 1.0, 5.0


def _style(ax, *, ylabel=None, xlabel=None, title=None, horizontal=False):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, axis="x" if horizontal else "y", alpha=0.2, linewidth=0.7)
    ax.tick_params(labelsize=8)
    if title:
        ax.set_title(title, fontsize=10.5, fontweight="bold", pad=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9)
    return ax


# ── individual plots ─────────────────────────────────────────────────────────────

def plot_marginals(evaluation: "Evaluation", *, ax=None):
    """Mean verdict per level of each factor (bars, 95% CI), coloured by factor."""
    import matplotlib.pyplot as plt

    df, factors = _frame(evaluation)
    lo, hi = _rubric_range(evaluation)
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 4))

    xs, heights, errs, colors, labels = [], [], [], [], []
    handles = []
    pos = 0
    for fi, f in enumerate(factors):
        g = df.groupby(f)["verdict"]
        means, sems = g.mean(), g.sem().fillna(0.0)
        color = PALETTE[fi % len(PALETTE)]
        handles.append((plt.Rectangle((0, 0), 1, 1, color=color), f))
        for lvl, label in zip(means.index, _labels(list(means.index))):
            xs.append(pos)
            heights.append(float(means[lvl]))
            errs.append(1.96 * float(sems[lvl]))
            colors.append(color)
            labels.append(label)
            pos += 1
        pos += 1  # gap between factor groups

    ax.bar(xs, heights, yerr=errs, capsize=3, color=colors, edgecolor="black", linewidth=0.5, alpha=0.92)
    for x, h in zip(xs, heights):
        ax.text(x, h, f"{h:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(lo, hi + 0.15)
    ax.legend([h for h, _ in handles], [n for _, n in handles], fontsize=8, title="factor", loc="lower right")
    return _style(ax, ylabel="mean verdict", title="Factor marginal means (95% CI)")


def plot_interaction(evaluation: "Evaluation", *, factors=None, ax=None):
    """Interaction plot: x = levels of factor A, one line per level of factor B.
    Non-parallel lines signal an interaction."""
    import matplotlib.pyplot as plt

    df, allf = _frame(evaluation)
    if len(allf) < 2:
        raise ValueError("interaction plot needs at least two factors")
    fa, fb = (factors or allf[:2])
    lo, hi = _rubric_range(evaluation)
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 4))

    a_levels = sorted(df[fa].dropna().unique(), key=str)
    b_levels = sorted(df[fb].dropna().unique(), key=str)
    for i, (b, b_label) in enumerate(zip(b_levels, _labels(b_levels))):
        sub = df[df[fb] == b]
        means = [sub.loc[sub[fa] == a, "verdict"].mean() for a in a_levels]
        ax.plot(range(len(a_levels)), means, marker="o", linewidth=2,
                color=PALETTE[i % len(PALETTE)], label=b_label)
    ax.set_xticks(range(len(a_levels)))
    ax.set_xticklabels(_labels(a_levels))
    ax.set_ylim(lo, hi + 0.15)
    ax.legend(fontsize=8, title=fb)
    return _style(ax, xlabel=fa, ylabel="mean verdict", title=f"Interaction: {fa} × {fb}")


def plot_configs(evaluation: "Evaluation", *, ax=None):
    """Every configuration ranked by mean verdict (horizontal bars)."""
    import matplotlib.pyplot as plt

    df, factors = _frame(evaluation)
    g = df.groupby(factors)["verdict"].mean().sort_values()
    labels = _cfg_labels(list(g.index))
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 0.5 * len(g) + 1.2))

    ax.barh(range(len(g)), g.values, color=ACCENT, edgecolor="black", linewidth=0.5, alpha=0.92)
    for i, v in enumerate(g.values):
        ax.text(v, i, f" {v:.2f}", va="center", fontsize=7)
    ax.set_yticks(range(len(g)))
    ax.set_yticklabels(labels)
    return _style(ax, xlabel="mean verdict",
                  title=f"Configurations ranked  ({' · '.join(factors)})", horizontal=True)


def plot_distribution(evaluation: "Evaluation", *, by=None, ax=None):
    """Verdict distribution (box plot) per configuration, or per factor if ``by`` is set."""
    import matplotlib.pyplot as plt

    df, factors = _frame(evaluation)
    key = [by] if isinstance(by, str) else (by or factors)
    groups = list(df.groupby(key)["verdict"])
    labels = _cfg_labels([g for g, _ in groups])
    data = [vals.values for _, vals in groups]
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 4))

    bp = ax.boxplot(data, patch_artist=True, widths=0.6, medianprops={"color": "black"})
    for patch in bp["boxes"]:
        patch.set(facecolor=ACCENT, alpha=0.45, edgecolor="black", linewidth=0.6)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    return _style(ax, ylabel="verdict", title=f"Verdict distribution  ({' · '.join(key)})")


def plot_effects(evaluation: "Evaluation", *, ax=None):
    """Effect-size forest, routed to the model that matches the rubric's scale — a
    **log-odds** forest for a binary rubric (consistent with the logistic GLMM in
    ``report()``), a **Cohen's d** forest otherwise."""
    from cafe.judging.rubric import ScaleType

    scale = getattr(getattr(getattr(evaluation, "ratings", None), "rubric", None), "scale_type", None)
    if scale == ScaleType.binary:
        return _plot_logodds_forest(evaluation, ax=ax)

    import matplotlib.pyplot as plt

    eff = evaluation.effects
    ds = [d for d in (eff.pairwise_d if eff else []) if d.get("d") is not None]
    if not ds:
        raise ValueError("no effect sizes to plot")
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 0.5 * len(ds) + 1.2))

    for y, d in enumerate(ds):
        ax.plot([d["ci_low"], d["ci_high"]], [y, y], color=_GREY, linewidth=1.6, zorder=2)
        ax.plot(d["d"], y, "o", color=ACCENT, markersize=8, markeredgecolor="black", zorder=3)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    levels: dict[str, list[Any]] = {}
    for d in ds:
        seen = levels.setdefault(d["factor"], [])
        seen += [x for x in (d["level_a"], d["level_b"]) if x not in seen]
    short = {f: dict(zip(lv, _labels(lv))) for f, lv in levels.items()}
    ax.set_yticks(range(len(ds)))
    ax.set_yticklabels([f"{d['factor']}: {short[d['factor']][d['level_a']]} vs "
                        f"{short[d['factor']][d['level_b']]}" for d in ds])
    return _style(ax, xlabel="Cohen's d  (gap between levels; 0=no effect)",
                  title="Effect sizes (95% CI)", horizontal=True)


def _plot_logodds_forest(evaluation: "Evaluation", *, ax=None):
    """Per-term log-odds forest for a binary rubric (from the logistic GLMM)."""
    import matplotlib.pyplot as plt

    log = evaluation.logistic
    terms = [t for t in (log.terms if log and log.available else []) if t.get("coef") is not None]
    if not terms:
        raise ValueError("no logistic effects to plot (binary rubric with no fitted terms)")
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 0.5 * len(terms) + 1.2))

    for y, t in enumerate(terms):
        ax.plot(t["coef"], y, "o", color=ACCENT, markersize=8, markeredgecolor="black", zorder=3)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_yticks(range(len(terms)))
    ax.set_yticklabels([t["label"] for t in terms])
    return _style(ax, xlabel="log-odds of a PASS  (+ = more likely; 0 = no effect)",
                  title="Effect sizes — logistic (log-odds)", horizontal=True)


def plot_pareto(evaluation: "Evaluation", *, ax=None):
    """Quality vs a resource objective (cost/latency/tokens), with the Pareto frontier."""
    from cafe.stats.pareto import pareto

    pf = pareto(evaluation)
    x = next((o for o in pf.objectives if o != "quality"), None)
    if x is None:
        raise ValueError("Pareto needs a varying cost/latency/tokens objective (none here — Mode A?)")
    return pf.plot(x=x, y="quality", ax=ax)


PLOTS = {
    "marginals": plot_marginals,
    "interaction": plot_interaction,
    "configs": plot_configs,
    "distribution": plot_distribution,
    "effects": plot_effects,
    "pareto": plot_pareto,
}


# ── dashboard + dispatcher ─────────────────────────────────────────────────────────

def dashboard(evaluation: "Evaluation"):
    """A grid of the key plots that apply to this evaluation."""
    import matplotlib.pyplot as plt

    _, factors = _frame(evaluation)
    panels = [("marginals", plot_marginals)]
    if len(factors) >= 2:
        panels.append(("interaction", plot_interaction))
    panels += [("configs", plot_configs), ("distribution", plot_distribution)]
    if evaluation.effects and any(d.get("d") is not None for d in evaluation.effects.pairwise_d):
        panels.append(("effects", plot_effects))
    try:
        from cafe.stats.pareto import pareto

        if any(o != "quality" for o in pareto(evaluation).objectives):
            panels.append(("pareto", plot_pareto))
    except Exception:  # noqa: BLE001 — pareto is optional in the dashboard
        pass

    cols = 2
    rows = (len(panels) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4.3 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for (name, fn), a in zip(panels, axes):
        try:
            fn(evaluation, ax=a)
        except Exception as exc:  # noqa: BLE001 — show which panel couldn't render
            a.axis("off")
            a.text(0.5, 0.5, f"{name}: n/a\n({exc})", ha="center", va="center", fontsize=8, color=_GREY)
    for a in axes[len(panels):]:
        a.axis("off")
    fig.suptitle(f"CAFE — {evaluation.study_name}", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def plot(evaluation: "Evaluation", kind: str | None = None, **kwargs):
    """Dispatch: ``kind=None`` → dashboard (Figure); a name → that single plot (Axes)."""
    if kind is None:
        return dashboard(evaluation)
    fn = PLOTS.get(kind)
    if fn is None:
        raise ValueError(f"unknown plot {kind!r}; choose from {sorted(PLOTS)} (or None for the dashboard)")
    return fn(evaluation, **kwargs)
