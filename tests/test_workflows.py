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


def test_phase6_7_8_use_streamed_artifact_lookup():
    for name in ("phase6.yml", "phase7.yml", "phase8.yml"):
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
