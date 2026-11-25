"""A small, causal formula language for LLM-proposed factors (phase 11).

The LLM writes text; this module decides what may run. A formula is
parsed with `ast` and evaluated by walking a whitelist, never `eval`:

  fields     close, open, volume, ret, bm, ep, roe, ebit_tev, gpa, roa
             (wide days x names panels; fundamentals are point-in-time,
             i.e. the value on day d was filed before d)
  operators  + - * /, unary -, and
             ts_mean(x,n) ts_std(x,n) ts_sum(x,n) ts_min(x,n) ts_max(x,n)
             ts_rank(x,n) delay(x,n) delta(x,n) ts_corr(x,y,n)
             rank(x) abs(x) log(x) sign(x)

Every time-series operator reads the trailing n rows ending at d, and
delay/delta need n >= 1, so a value on day d depends only on rows <= d.
Windows are integer literals in [1, 252] (delay/delta) or [2, 252].
Formulas are capped at MAX_NODES syntax nodes (AlphaAgent-style limit).
"""
from __future__ import annotations

import ast

import numpy as np
import pandas as pd

FIELDS = ("close", "open", "volume", "ret", "bm", "ep", "roe", "ebit_tev", "gpa", "roa")
TS1 = {"ts_mean", "ts_std", "ts_sum", "ts_min", "ts_max", "ts_rank", "delay", "delta"}
CS1 = {"rank", "abs", "log", "sign"}
MAX_NODES = 30
MAX_WIN = 252


class FormulaError(ValueError):
    pass


def _window(node: ast.AST, lo: int) -> int:
    if not (isinstance(node, ast.Constant) and isinstance(node.value, int)
            and not isinstance(node.value, bool)):
        raise FormulaError("window must be an integer literal")
    if not lo <= node.value <= MAX_WIN:
        raise FormulaError(f"window {node.value} outside [{lo}, {MAX_WIN}]")
    return node.value


def _check(node: ast.AST) -> set[str]:
    """Validate one node recursively; returns the fields it reads."""
    if isinstance(node, ast.Name):
        if node.id not in FIELDS:
            raise FormulaError(f"unknown field {node.id!r}")
        return {node.id}
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise FormulaError("only numeric constants")
        return set()
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return _check(node.operand)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        return _check(node.left) | _check(node.right)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
        f, a = node.func.id, node.args
        if f in TS1 and len(a) == 2:
            _window(a[1], 1 if f in ("delay", "delta") else 2)
            return _check(a[0])
        if f == "ts_corr" and len(a) == 3:
            _window(a[2], 2)
            return _check(a[0]) | _check(a[1])
        if f in CS1 and len(a) == 1:
            return _check(a[0])
        raise FormulaError(f"bad call {f}/{len(a)}")
    raise FormulaError(f"disallowed syntax {type(node).__name__}")


def parse(expr: str) -> ast.Expression:
    """Parse + validate; raises FormulaError on anything outside the language."""
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as e:
        raise FormulaError(f"syntax: {e.msg}") from None
    if sum(1 for _ in ast.walk(tree)) - 1 > MAX_NODES:
        raise FormulaError(f"more than {MAX_NODES} nodes")
    if not _check(tree.body):
        raise FormulaError("formula reads no field")
    return tree


def canonical(expr: str) -> str:
    """Whitespace/format-insensitive form used to de-duplicate proposals."""
    return ast.unparse(parse(expr).body)


def _ts(x: pd.DataFrame, f: str, n: int) -> pd.DataFrame:
    if f == "delay":
        return x.shift(n)
    if f == "delta":
        return x - x.shift(n)
    r = x.rolling(n, min_periods=n)
    return {"ts_mean": r.mean, "ts_std": r.std, "ts_sum": r.sum, "ts_min": r.min,
            "ts_max": r.max, "ts_rank": lambda: r.rank(pct=True)}[f]()


def _eval(node: ast.AST, env: dict[str, pd.DataFrame]):
    if isinstance(node, ast.Name):
        return env[node.id]
    if isinstance(node, ast.Constant):
        return float(node.value)
    if isinstance(node, ast.UnaryOp):
        return -_eval(node.operand, env)
    if isinstance(node, ast.BinOp):
        a, b = _eval(node.left, env), _eval(node.right, env)
        if isinstance(node.op, ast.Add):
            return a + b
        if isinstance(node.op, ast.Sub):
            return a - b
        if isinstance(node.op, ast.Mult):
            return a * b
        with np.errstate(divide="ignore", invalid="ignore"):
            out = a / b
        return out.replace([np.inf, -np.inf], np.nan) if isinstance(out, pd.DataFrame) else out
    f, a = node.func.id, node.args
    x = _eval(a[0], env)
    if not isinstance(x, pd.DataFrame):
        raise FormulaError(f"{f} needs a panel argument")
    if f in TS1:
        return _ts(x, f, a[1].value)
    if f == "ts_corr":
        y = _eval(a[1], env)
        return x.rolling(a[2].value, min_periods=a[2].value).corr(y) \
            .replace([np.inf, -np.inf], np.nan)
    if f == "rank":
        return x.rank(axis=1, pct=True)
    if f == "abs":
        return x.abs()
    if f == "log":
        return np.log(x.where(x > 0))
    return np.sign(x)


def evaluate(expr: str, env: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Formula -> days x names panel (NaN where undefined)."""
    out = _eval(parse(expr).body, env)
    if not isinstance(out, pd.DataFrame):
        raise FormulaError("formula is a constant")
    return out
