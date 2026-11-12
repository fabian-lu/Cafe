"""`cafe run` / `cafe validate` import a study file the way `python study.py` runs it.

The file is registered in sys.modules (dataclasses look a class's module up there) and its
directory is first on sys.path while the command runs, so it can import the modules next to it
(the case studies are split into such sibling modules) — also lazily, from inside the system.
"""

import os
import subprocess
import sys
import textwrap

import pytest

import cafe
from cafe import cli

SIBLINGS = ("sib_helpers_load", "sib_helpers_lazy")


@pytest.fixture(autouse=True)
def _forget_study_modules():
    """Unload what these tests import, so a later test can't pick up a stale copy."""
    yield
    for name in ("cafe_user_study", *SIBLINGS):
        sys.modules.pop(name, None)


def _write(path, source):
    path.write_text(textwrap.dedent(source))
    return path


def _elsewhere(tmp_path, monkeypatch):
    """Run from a directory that is neither the study's nor on sys.path."""
    cwd = tmp_path / "elsewhere"
    cwd.mkdir()
    monkeypatch.chdir(cwd)


def test_study_with_future_annotations_and_a_dataclass(tmp_path, monkeypatch, capsys):
    study = _write(tmp_path / "study_dc.py", """\
        from __future__ import annotations

        from dataclasses import dataclass

        import cafe


        @dataclass
        class System:
            prefix: str = "ans"

            async def run(self, config, item):
                return f"{self.prefix}:{config['model']}:{item}"


        study = cafe.Study(name="dc", system=System(), factors=[cafe.Factor("model", ["a", "b"])],
                           dataset=["q1", "q2"])
        """)
    _elsewhere(tmp_path, monkeypatch)
    assert cli.main(["validate", str(study)]) == 0  # used to crash inside dataclasses
    assert "study 'dc' is valid" in capsys.readouterr().out


def test_study_imports_a_module_next_to_it(tmp_path, monkeypatch, capsys):
    _write(tmp_path / "sib_helpers_load.py", """\
        def answer(config, item):
            return f"{config['model']}:{item}"
        """)
    study = _write(tmp_path / "study_sibling.py", """\
        import cafe
        from sib_helpers_load import answer

        study = cafe.Study(name="sib", system=answer, factors=[cafe.Factor("model", ["a", "b"])],
                           dataset=["q1", "q2"])
        """)
    _elsewhere(tmp_path, monkeypatch)
    path_before = sys.path[:]
    assert cli.main(["validate", str(study)]) == 0  # used to raise ModuleNotFoundError
    assert "study 'sib' is valid" in capsys.readouterr().out
    assert sys.path == path_before  # the study's directory is gone again


def test_system_imports_a_sibling_lazily_during_the_run(tmp_path, monkeypatch, capsys):
    _write(tmp_path / "sib_helpers_lazy.py", """\
        def answer(config, item):
            return f"lazy:{config['model']}:{item}"
        """)
    study = _write(tmp_path / "study_lazy.py", """\
        import cafe


        async def system(config, item):
            from sib_helpers_lazy import answer  # imported only once the run is under way

            return answer(config, item)


        study = cafe.Study(name="lazy", system=system, factors=[cafe.Factor("model", ["a", "b"])],
                           dataset=["q1"])
        """)
    out = tmp_path / "results.jsonl"
    _elsewhere(tmp_path, monkeypatch)
    assert cli.main(["run", str(study), "--out", str(out)]) == 0  # no cell errored
    assert "0 errors" in capsys.readouterr().out
    assert '"lazy:a:q1"' in out.read_text()


def test_a_failing_study_file_is_not_left_in_sys_modules(tmp_path, monkeypatch):
    study = _write(tmp_path / "broken.py", "raise RuntimeError('broken study')\n")
    _elsewhere(tmp_path, monkeypatch)
    path_before = sys.path[:]
    with pytest.raises(RuntimeError, match="broken study"):
        cli.main(["validate", str(study)])
    assert "cafe_user_study" not in sys.modules
    assert sys.path == path_before


def test_cafe_validate_in_a_fresh_interpreter(tmp_path):
    """The reported command: a dataclass study that imports a sibling, run like the `cafe`
    script, whose sys.path[0] is its bin directory rather than the study's."""
    _write(tmp_path / "sib_helpers_load.py", "PREFIX = 'ans'\n")
    study = _write(tmp_path / "study.py", """\
        from __future__ import annotations

        from dataclasses import dataclass

        import cafe
        from sib_helpers_load import PREFIX


        @dataclass
        class System:
            async def run(self, config, item):
                return f"{PREFIX}:{config['model']}:{item}"


        study = cafe.Study(name="both", system=System(), factors=[cafe.Factor("model", ["a"])],
                           dataset=["q1"])
        """)
    cwd = tmp_path / "elsewhere"
    cwd.mkdir()
    src = os.path.dirname(os.path.dirname(cafe.__file__))
    proc = subprocess.run(
        [sys.executable, "-c", "import sys; from cafe.cli import main; sys.exit(main())",
         "validate", str(study)],
        cwd=cwd, env={**os.environ, "PYTHONPATH": src}, capture_output=True, text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert "study 'both' is valid" in proc.stdout
