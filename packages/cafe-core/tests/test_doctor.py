"""`cafe doctor` and the R checks behind it (check_r / check_glmer).

clmm.R and glmm.R need the R package jsonlite, which neither ordinal nor lme4 depends on, so the
checks must ask for it and name exactly the package that is missing. And doctor only says "All set"
once the Python extras the statistics layer needs are installed too.
"""

import importlib.machinery
import importlib.util
import os
import shutil
import subprocess

import pytest

from cafe import cli
from cafe.stats import logistic, ordinal

EXTRAS = ("pandas", "numpy", "statsmodels", "scipy", "matplotlib", "litellm")


@pytest.fixture
def fake_r(monkeypatch):
    """An R on the PATH with ``ordinal`` and ``lme4`` but not ``jsonlite`` — what the old
    ``install.packages(c("ordinal", "lme4"))`` leaves. Add to the returned set to install more."""
    installed = {"ordinal", "lme4"}
    real_run = subprocess.run

    def run(cmd, *args, **kwargs):
        if list(cmd[:1]) == ["Rscript"]:  # doctor's R.version.string query
            return subprocess.CompletedProcess(cmd, 0, stdout="R version 4.5.0 (fake)", stderr="")
        return real_run(cmd, *args, **kwargs)

    def which(name, *args, **kwargs):
        return "/fake/bin/Rscript" if name == "Rscript" else None

    def missing_r_packages(packages):
        return [p for p in packages if p not in installed]

    monkeypatch.setattr(shutil, "which", which)
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(ordinal, "_missing_r_packages", missing_r_packages)
    return installed


def _extras(monkeypatch, missing=()):
    """Make every Python extra look installed, except the modules in ``missing``."""
    real = importlib.util.find_spec

    def find_spec(name, *args, **kwargs):
        if name in missing:
            return None
        if name in EXTRAS:
            return importlib.machinery.ModuleSpec(name, None)
        return real(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)


def _doctor(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)  # so main()'s load_env() finds no .env
    rc = cli.main(["doctor"])
    return rc, capsys.readouterr().out


def test_check_r_and_check_glmer_name_the_missing_jsonlite(fake_r):
    ok, msg = ordinal.check_r()
    assert not ok
    assert "'jsonlite'" in msg and "'ordinal'" not in msg
    assert "install.packages(\"jsonlite\")" in msg
    ok, msg = logistic.check_glmer()
    assert not ok
    assert "'jsonlite'" in msg and "lme4" not in msg  # it used to blame lme4

    fake_r.add("jsonlite")
    assert ordinal.check_r()[0] and logistic.check_glmer()[0]


def test_missing_r_packages_reads_rs_answer(monkeypatch):
    calls = []

    def run(cmd, *args, **kwargs):
        calls.append(cmd)
        stdout = "Rprofile noise\ncafe-missing: jsonlite \n"
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    assert ordinal._missing_r_packages(("ordinal", "jsonlite")) == ["jsonlite"]
    assert 'c("ordinal", "jsonlite")' in calls[0][-1]


def test_check_r_reports_an_r_that_does_not_answer(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name, *a, **k: "/fake/bin/" + name)
    monkeypatch.setattr(subprocess, "run", lambda cmd, *a, **k: subprocess.CompletedProcess(
        cmd, 2, stdout="", stderr="Fatal error: R home directory is not defined"))
    ok, msg = ordinal.check_r()
    assert not ok
    assert "failed to run" in msg and "R home directory is not defined" in msg  # not "missing"


def test_doctor_names_the_missing_jsonlite(fake_r, monkeypatch, tmp_path, capsys):
    _extras(monkeypatch)
    rc, out = _doctor(monkeypatch, tmp_path, capsys)
    assert rc == 1
    assert "✓  ordinal" in out and "✓  lme4" in out
    assert "✗  jsonlite" in out
    assert "fix →  Rscript -e 'install.packages(\"jsonlite\")'" in out
    assert "install.packages(\"lme4\")" not in out
    assert "All set" not in out


def test_doctor_without_r_installs_all_three_packages(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(shutil, "which", lambda name, *a, **k: None)
    rc, out = _doctor(monkeypatch, tmp_path, capsys)
    assert rc == 1
    assert "R not found" in out
    assert "Rscript -e 'install.packages(c(\"ordinal\", \"lme4\", \"jsonlite\"))'" in out


def test_doctor_all_set_with_r_and_every_extra(fake_r, monkeypatch, tmp_path, capsys):
    fake_r.add("jsonlite")
    _extras(monkeypatch)
    rc, out = _doctor(monkeypatch, tmp_path, capsys)
    assert rc == 0
    assert "✓  stats" in out and "✓  llm" in out and "✓  jsonlite" in out
    assert "All set" in out


def test_doctor_needs_the_stats_extra(fake_r, monkeypatch, tmp_path, capsys):
    fake_r.add("jsonlite")
    _extras(monkeypatch, missing=("pandas", "statsmodels"))
    rc, out = _doctor(monkeypatch, tmp_path, capsys)
    assert rc == 1
    assert "✗  stats  missing pandas, statsmodels" in out
    assert 'fix →  pip install "cafe-core[stats]"' in out
    assert "All set" not in out


def test_doctor_warns_without_litellm(fake_r, monkeypatch, tmp_path, capsys):
    # LLM calls need litellm, but the statistics and execution layers don't: a warning, exit 0.
    fake_r.add("jsonlite")
    _extras(monkeypatch, missing=("litellm",))
    rc, out = _doctor(monkeypatch, tmp_path, capsys)
    assert rc == 0
    assert "!  llm    missing litellm" in out
    assert 'fix →  pip install "cafe-core[llm]"' in out
    assert "All set" not in out
    assert "see the items marked ! above" in out


def test_real_r_without_jsonlite(monkeypatch, tmp_path, capsys):
    """Against the real R: shadow jsonlite with a package R can't load (as if it were missing)."""
    if not (shutil.which("Rscript") and ordinal.check_r()[0] and logistic.check_glmer()[0]):
        pytest.skip("needs R with the ordinal, lme4 and jsonlite packages")
    mask = tmp_path / "Rlib-mask" / "jsonlite"
    mask.mkdir(parents=True)
    (mask / "DESCRIPTION").write_text("Package: jsonlite\nVersion: 0.0.0\n")
    monkeypatch.setenv("R_LIBS", os.pathsep.join(filter(None, [str(mask.parent),
                                                              os.environ.get("R_LIBS")])))

    ok, msg = ordinal.check_r()
    assert not ok and "'jsonlite'" in msg
    ok, msg = logistic.check_glmer()
    assert not ok and "'jsonlite'" in msg and "lme4" not in msg

    rc, out = _doctor(monkeypatch, tmp_path, capsys)
    assert rc == 1
    assert "✗  jsonlite" in out and "✗  lme4" not in out and "✗  ordinal" not in out
