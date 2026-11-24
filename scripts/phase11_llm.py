#!/usr/bin/env python3
"""Phase-11 LLM layer, part 1 — a local open-weights model proposes factors.

    python scripts/phase11_llm.py propose   # -> results/phase11/llm_factors.json

Heavy (a 1.5B model on CPU) — GitHub Actions only.

The model gets the formula language of src/quant_research/factor_dsl.py
and a theme. It never sees a ticker, a date or a return. Its text is
parsed and validated by factor_dsl; anything outside the language is
rejected with the reason recorded. The first `max_factors` distinct valid
formulas, in generation order, are the phase-11 LLM factor set. They are
committed before the dev run and are not edited afterwards. Testing,
weighting and sizing are done by the algorithm (scripts/phase11_us.py).
"""
from __future__ import annotations

import datetime as dt
import json
import re
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_research import factor_dsl as D  # noqa: E402

SPEC_PATH = REPO / "configs" / "experiments" / "phase11_us.yaml"
OUT = REPO / "results" / "phase11" / "llm_factors.json"

SYSTEM = "You are a careful quantitative equity researcher."
PROMPT = """Propose {k} cross-sectional stock-selection factors for large US
companies. A higher factor value must mean a higher expected return over
the next month. Theme: {theme}.

Write each factor with ONLY this formula language:
- fields (one value per stock per day): close, open, volume, ret (daily
  return), bm (book/market), ep (earnings/price), roe (return on equity),
  ebit_tev (operating income / enterprise value), gpa (gross profit /
  assets), roa (return on assets)
- operators: + - * / and parentheses, numbers
- ts_mean(x, n), ts_std(x, n), ts_sum(x, n), ts_min(x, n), ts_max(x, n),
  ts_rank(x, n), delay(x, n), delta(x, n), ts_corr(x, y, n): over the past
  n trading days, n a whole number from 2 to 252 (delay/delta from 1)
- rank(x), abs(x), log(x), sign(x)

Syntax example (do not reuse it): -ts_std(ret, 21) / ts_mean(volume, 5)

Answer with exactly {k} lines and nothing else, each line:
short_name | formula | one-sentence economic reason"""


def log(msg: str) -> None:
    print(f"[phase11-llm {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_lines(text: str) -> list[tuple[str, str, str]]:
    """'name | formula | reason' lines -> tuples (other lines ignored)."""
    out = []
    for ln in text.splitlines():
        parts = [p.strip().strip("`") for p in ln.split("|")]
        if len(parts) >= 2 and parts[1]:
            name = re.sub(r"^\s*[\d\.\)\-\*]+\s*", "", parts[0])
            out.append((name, parts[1], parts[2] if len(parts) > 2 else ""))
    return out


def select(raw: list[dict], max_factors: int) -> tuple[list[dict], list[dict]]:
    """Validate + de-duplicate in generation order."""
    keep, rejected, seen = [], [], set()
    for r in raw:
        for name, formula, why in parse_lines(r["text"]):
            try:
                canon = D.canonical(formula)
            except D.FormulaError as e:
                rejected.append({"theme": r["theme"], "formula": formula, "reason": str(e)})
                continue
            if canon in seen:
                rejected.append({"theme": r["theme"], "formula": formula, "reason": "duplicate"})
                continue
            if len(keep) >= max_factors:
                rejected.append({"theme": r["theme"], "formula": formula, "reason": "over cap"})
                continue
            seen.add(canon)
            keep.append({"id": f"L{len(keep) + 1:02d}", "name": name, "formula": canon,
                         "reason": why, "theme": r["theme"]})
    return keep, rejected


def cmd_propose(spec: dict) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = spec["llm"]
    tok = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revision"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], revision=cfg["revision"], torch_dtype=torch.float32).eval()
    raw = []
    for i, theme in enumerate(cfg["themes"]):
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": PROMPT.format(k=cfg["per_theme"], theme=theme)}]
        x = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
        torch.manual_seed(cfg["seed"] + i)
        t0 = time.time()
        with torch.no_grad():
            y = model.generate(x, max_new_tokens=cfg["max_new_tokens"], do_sample=True,
                               temperature=cfg["temperature"], top_p=cfg["top_p"])
        text = tok.decode(y[0, x.shape[1]:], skip_special_tokens=True)
        raw.append({"theme": theme, "seed": cfg["seed"] + i, "text": text})
        log(f"theme {i + 1}/{len(cfg['themes'])} ({time.time() - t0:.0f}s):\n{text}")
    keep, rejected = select(raw, cfg["max_factors"])
    out = {"model": cfg["model"], "revision": cfg["revision"],
           "generated_at": dt.datetime.now().isoformat(),
           "prompt": PROMPT, "system": SYSTEM, "settings": cfg,
           "raw": raw, "factors": keep, "rejected": rejected}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    log(f"{len(keep)} factors kept, {len(rejected)} rejected -> {OUT}")
    return out


def main() -> None:
    if sys.argv[1:] != ["propose"]:
        sys.exit("usage: phase11_llm.py propose")
    cmd_propose(yaml.safe_load(SPEC_PATH.read_text()))


if __name__ == "__main__":
    main()
