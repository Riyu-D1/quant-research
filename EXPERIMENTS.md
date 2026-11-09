# Experiment log

Every run counts — including failures and reproduce-only runs. Numbering is
monotonic; `EX-000` is environment setup. "Δ" columns are vs the best published
baseline for the same dataset/features once Phase 2 starts.

| ID | Date | Phase | Change | Dataset | IC | ICIR | Rank IC | Ann. ret | IR | Max DD | Turnover | Verdict |
|----|------|-------|--------|---------|----|------|---------|----------|----|--------|----------|---------|
| EX-000 | 2026-11-09 | 1 | Repo scaffold, py3.11 + uv, pinned deps | — | — | — | — | — | — | — | — | setup |
| EX-001 | 2026-11-09 | 1 | LightGBM Alpha158 CSI300 reproduction (published IC 0.0448 / IR 1.0164 / MDD -0.1038) | csi300 | 0.0468 | 0.3816 | 0.0490 | 0.0807 | 0.9145 | -0.0861 | 0.1911 | reproduced, close to published (v2 data) |

## Running total of hypotheses tested

0 (baseline reproduction does not count as a hypothesis).
