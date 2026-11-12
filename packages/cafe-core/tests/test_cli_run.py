"""`cafe run --smoke` prints the full run's time/cost estimate, and `cafe run` (answers only) says
so when the study has a rubric and a judge."""

import textwrap

from cafe import cli


def _run(monkeypatch, tmp_path, capsys, *argv):
    monkeypatch.chdir(tmp_path)  # so main()'s load_env() finds no .env
    rc = cli.main(["run", *argv])
    return rc, capsys.readouterr().out


def test_smoke_prints_the_full_run_estimate(monkeypatch, tmp_path, capsys):
    rc, out = _run(monkeypatch, tmp_path, capsys, "example", "--smoke")
    assert rc == 0
    # the toy study: 4 configs x 4 inputs x 3 reps, and its system reports a cost per answer
    assert "full run (3 reps): 48 cells; est. compute ~" in out
    assert "est. cost ~$" in out


def test_smoke_estimate_counts_the_requested_reps(monkeypatch, tmp_path, capsys):
    rc, out = _run(monkeypatch, tmp_path, capsys, "example", "--smoke", "--reps", "5")
    assert rc == 0
    assert "full run (5 reps): 80 cells" in out


def test_smoke_estimate_without_a_reported_cost(monkeypatch, tmp_path, capsys):
    study = tmp_path / "plain.py"
    study.write_text(textwrap.dedent("""\
        import cafe

        study = cafe.Study(name="plain", system=lambda config, item: f"answer {item}",
                           factors=[cafe.Factor("model", ["a", "b"])], dataset=["q1", "q2"],
                           replications=2)
        """))
    rc, out = _run(monkeypatch, tmp_path, capsys, str(study), "--smoke")
    assert rc == 0
    assert "full run (2 reps): 8 cells; est. compute ~" in out
    assert "est. cost n/a (the system reports no cost_usd)" in out


def test_full_run_prints_no_estimate(monkeypatch, tmp_path, capsys):
    rc, out = _run(monkeypatch, tmp_path, capsys, "example", "--reps", "1")
    assert rc == 0
    assert "est. compute" not in out


def test_run_says_it_does_not_judge(monkeypatch, tmp_path, capsys):
    study = tmp_path / "judged.py"
    study.write_text(textwrap.dedent("""\
        import cafe


        class NeverCalled:
            model = "never"

            async def score(self, *args, **kwargs):
                raise AssertionError("cafe run must not judge")


        study = cafe.Study(name="judged", system=lambda config, item: f"answer {item}",
                           factors=[cafe.Factor("model", ["a", "b"])], dataset=["q1"],
                           rubric=cafe.ANSWER_QUALITY_1_5, judge=NeverCalled())
        """))
    rc, out = _run(monkeypatch, tmp_path, capsys, str(study))
    assert rc == 0
    assert "judging : skipped" in out and "study.evaluate()" in out
