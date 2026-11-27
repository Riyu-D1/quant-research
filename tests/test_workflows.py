"""Workflow hygiene — static checks on .github/workflows (no network).

`gh api --paginate -q FILTER` runs FILTER once per page, so an array
reduction such as `[...] | max` prints one value per page and breaks the
caller once the repo has more than one page of artifacts ("accepts 1
arg(s), received 2"). Lookups must stream ids and reduce in the shell:
`... | .id' | sort -n | tail -1`.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

WF = Path(__file__).resolve().parents[1] / ".github" / "workflows"
REDUCERS = re.compile(r"\]\s*\|\s*(max|min|first|last|length|add)\b")


def _paginated_filters(text: str) -> list[str]:
    """The -q filter of every `gh api ... --paginate` command (line
    continuations joined)."""
    joined = re.sub(r"\\\n\s*", " ", text)
    out = []
    for line in joined.splitlines():
        if "gh api" in line and "--paginate" in line:
            m = re.search(r"-q\s+(\"(?:[^\"\\]|\\.)*\"|'[^']*')", line)
            if m:
                out.append(m.group(1))
    return out


def test_all_workflows_parse():
    files = sorted(WF.glob("*.yml"))
    assert files
    for f in files:
        assert isinstance(yaml.safe_load(f.read_text()), dict), f.name


def test_paginated_lookups_do_not_reduce_per_page():
    bad = {f.name: q for f in WF.glob("*.yml")
           for q in _paginated_filters(f.read_text()) if REDUCERS.search(q)}
    assert not bad, f"per-page reduction under --paginate: {bad}"


def test_artifact_lookups_are_streamed_and_paginated():
    for name in ("phase4.yml", "phase6.yml", "phase7.yml", "phase8.yml",
                 "paper_trade.yml"):
        text = (WF / name).read_text()
        qs = _paginated_filters(text)
        assert qs, name
        assert "| sort -n | tail -1)" in text, name
        assert '[ -n "$AID" ]' in text, name


def test_filter_detector_catches_the_old_form():
    old = ('AID=$(gh api repos/x/actions/artifacts --paginate \\\n'
           '  -q "[.artifacts[] | select(.name==\\"$1\\") | .id] | max")')
    assert any(REDUCERS.search(q) for q in _paginated_filters(old))
    new = ('AID=$(gh api repos/x/actions/artifacts --paginate \\\n'
           '  -q ".artifacts[] | select(.name==\\"$1\\") | .id" | sort -n | tail -1)')
    assert not any(REDUCERS.search(q) for q in _paginated_filters(new))


def test_pytest_workflow_runs_on_pull_requests():
    wf = yaml.safe_load((WF / "pytest.yml").read_text())
    on = wf.get("on", wf.get(True))       # PyYAML reads bare `on` as True
    assert "pull_request" in on
    steps = " ".join(str(s.get("run", "")) for s in wf["jobs"]["test"]["steps"])
    assert "pytest tests/" in steps


def test_every_artifact_list_call_paginates():
    """An unpaginated `gh api .../actions/artifacts -q` sees only the newest
    30 artifacts and silently misses older ones."""
    bad = []
    for f in WF.glob("*.yml"):
        joined = re.sub(r"\\\n\s*", " ", f.read_text())
        bad += [f"{f.name}: {ln.strip()[:80]}" for ln in joined.splitlines()
                if re.search(r"gh api .*?/actions/artifacts(\s|$)", ln)
                and "--paginate" not in ln]
    assert not bad, bad


def test_qv50_paper_job_is_independent_and_uses_separate_paper_keys():
    wf = yaml.safe_load((WF / "paper_trade.yml").read_text())
    jobs = wf["jobs"]
    assert "run" in jobs and "run_qv50" in jobs
    assert "needs" not in jobs["run"] and "needs" not in jobs["run_qv50"]
    assert jobs["run"]["env"]["ALPACA_API_KEY"] == "${{ secrets.ALPACA_API_KEY }}"
    qv_env = jobs["run_qv50"]["env"]
    assert qv_env["ALPACA_API_KEY"] == "${{ secrets.ALPACA_QV50_KEY }}"
    assert qv_env["ALPACA_SECRET_KEY"] == "${{ secrets.ALPACA_QV50_SECRET }}"
    steps = jobs["run_qv50"]["steps"]
    paper = next(s for s in steps if s.get("name") == "qv50 paper day")
    args = paper["env"]["ARGS"]
    assert "--config configs/qv50_paper.yaml" in args
    assert "--rec-dir results/paper_run_qv50" in args
    assert "inputs.config" not in args
    qv = yaml.safe_load((WF.parents[1] / "configs" / "qv50_paper.yaml").read_text())
    assert qv["validated"] is False and qv["live_trading"] is False
    assert qv["strategy"] == "QV50"
    assert qv["signals"] == ["qv"]
    assert qv["core"] == {"symbol": "SPY", "weight": 0.5}
    assert qv["sleeve"]["n"] == 20
    assert qv["rebalance_band"] == 0.02
    assert qv["risk"]["max_position_weight"] == 0.2
    assert qv["membership_csv"] == "results/phase11/sp500_membership.csv"
