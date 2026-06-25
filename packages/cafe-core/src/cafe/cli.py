"""``cafe`` command-line interface.

Slice 1 commands:

- ``cafe run example``          run the bundled toy study
- ``cafe run path/to/study.py`` run a study defined in a Python file
- ``cafe validate [target]``    expand the design and report its size (no execution)
- ``cafe version``

Flags for ``run``: ``--smoke`` (preflight), ``--reps``, ``--concurrency``,
``--checkpoint PATH`` (resumable), ``--out PATH`` (write results JSONL).

The CLI is the headless/CI path; it shares the exact engine the library and (later)
the web platform use.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from cafe import design, run_study
from cafe.execution.results import Observation, Results, config_label
from cafe.study import Study


def _load_study(target: str) -> Study:
    """Resolve a CLI target into a Study.

    ``example`` -> the bundled toy study. Otherwise ``target`` is a path to a
    .py file exposing a module-level ``study`` or a ``build_study()``/
    ``build_example_study()`` callable.
    """
    if target == "example":
        from cafe.examples import build_example_study

        return build_example_study()

    path = Path(target)
    if not path.exists():
        raise SystemExit(f"cafe: no such study target: {target!r}")

    spec = importlib.util.spec_from_file_location("cafe_user_study", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cafe: could not import {target!r}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for attr in ("study", "STUDY"):
        obj = getattr(module, attr, None)
        if isinstance(obj, Study):
            return obj
    for attr in ("build_study", "build_example_study"):
        fn = getattr(module, attr, None)
        if callable(fn):
            obj = fn()
            if isinstance(obj, Study):
                return obj
    raise SystemExit(
        f"cafe: {target!r} must define a `study` (Study) or a `build_study()` callable"
    )


def _progress(_obs: Observation, done: int, total: int) -> None:
    bar_w = 24
    filled = int(bar_w * done / total) if total else bar_w
    bar = "█" * filled + "·" * (bar_w - filled)
    sys.stderr.write(f"\r  running [{bar}] {done}/{total}")
    sys.stderr.flush()
    if done == total:
        sys.stderr.write("\n")


def _mean_meta(obs: list[Observation], key: str) -> float | None:
    vals = [o.metadata[key] for o in obs if key in o.metadata and o.metadata[key] is not None]
    return round(statistics.fmean(vals), 3) if vals else None


def _print_results_table(results: Results) -> None:
    by_config: dict[str, list[Observation]] = defaultdict(list)
    for o in results.observations:
        by_config[config_label(o.config)].append(o)

    rows: list[tuple[str, ...]] = []
    header = ("configuration", "n", "err", "lat_s", "cost$", "sim_q")
    for label in sorted(by_config):
        obs = by_config[label]
        n_err = sum(1 for o in obs if not o.ok)
        lat = _mean_meta(obs, "latency_s")
        cost = _mean_meta(obs, "cost_usd")
        simq = _mean_meta(obs, "sim_quality")
        rows.append(
            (
                label,
                str(len(obs)),
                str(n_err),
                "-" if lat is None else f"{lat:.3f}",
                "-" if cost is None else f"{cost:.4f}",
                "-" if simq is None else f"{simq:.3f}",
            )
        )

    widths = [max(len(header[i]), *(len(r[i]) for r in rows)) for i in range(len(header))]
    fmt = "  ".join("{:<" + str(w) + "}" for w in widths)
    print()
    print("  " + fmt.format(*header))
    print("  " + fmt.format(*("-" * w for w in widths)))
    for r in rows:
        print("  " + fmt.format(*r))
    print()


def _cmd_run(args: argparse.Namespace) -> int:
    study = _load_study(args.target)
    n_inputs = 1 if args.smoke else max(1, len(study.dataset))
    n_reps = 1 if args.smoke else (args.reps or study.replications)
    n_cells = design.size(study) * n_inputs * n_reps
    mode = "SMOKE (1 input, 1 rep)" if args.smoke else f"{n_reps} reps"
    print(f"study   : {study.name}")
    print(f"factors : {', '.join(f'{f.name}{f.levels}' for f in study.factors) or '(none)'}")
    print(f"design  : {study.design} -> {design.size(study)} configs")
    print(f"inputs  : {len(study.dataset)}   mode: {mode}")
    print(f"cells   : {n_cells}")

    results = asyncio.run(
        run_study(
            study,
            replications=args.reps,
            concurrency=args.concurrency,
            checkpoint_path=args.checkpoint,
            smoke=args.smoke,
            on_progress=_progress,
        )
    )

    s = results.summary()
    _print_results_table(results)
    print(
        f"  done: {s['n_answers']} obs over {s['n_configs']} configs, "
        f"{s['n_errors']} errors, {s['total_compute_s']}s compute"
    )
    if args.out:
        results.to_jsonl(args.out)
        print(f"  wrote {args.out}")
    return 1 if s["n_errors"] else 0


def _cmd_validate(args: argparse.Namespace) -> int:
    study = _load_study(args.target)
    configs = design.generate(study)
    print(f"ok: study {study.name!r} is valid")
    print(f"    design  : {study.design} -> {len(configs)} configurations")
    print(f"    factors : {len(study.factors)}   inputs: {len(study.dataset)}")
    for c in configs[: min(8, len(configs))]:
        print(f"      - {config_label(c)}")
    if len(configs) > 8:
        print(f"      ... (+{len(configs) - 8} more)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cafe", description=__doc__)
    parser.add_argument("--version", action="store_true", help="print version and exit")
    sub = parser.add_subparsers(dest="command")

    p_run = sub.add_parser("run", help="run a study (or `example`)")
    p_run.add_argument("target", help="'example' or a path to a .py study file")
    p_run.add_argument("--smoke", action="store_true", help="preflight: 1 input, 1 rep, no judging")
    p_run.add_argument("--reps", type=int, default=None, help="replications per (config, input)")
    p_run.add_argument("--concurrency", type=int, default=8, help="max cells in flight")
    p_run.add_argument("--checkpoint", default=None, help="resumable checkpoint JSONL path")
    p_run.add_argument("--out", default=None, help="write results to this JSONL path")
    p_run.set_defaults(func=_cmd_run)

    p_val = sub.add_parser("validate", help="expand a study's design without running it")
    p_val.add_argument("target", nargs="?", default="example", help="'example' or a .py study file")
    p_val.set_defaults(func=_cmd_validate)

    args = parser.parse_args(argv)
    if args.version:
        from cafe import __version__

        print(f"cafe {__version__}")
        return 0
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
