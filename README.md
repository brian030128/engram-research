# Engram research (Issue #277)

| Deliverable | Path |
|---|---|
| Research report (Markdown, zh-TW) | `report/Engram_研究報告.md` |
| Architecture diagram | `report/figures/engram_dataflow.svg`, `.png` |
| Charts | `report/figures/` |
| Next-phase experiment plan | Report section 5.5 |
| Measured results | `results/demo/`, `results/expA/` |
| Scripts | `scripts/` |
| Sharing deck, English (42 slides, ordered by issue #277 items) | `slides/project/` |
| Sharing deck, Traditional Chinese (20 slides) | `slides_zh/project/` |
| Notes and external sources | `notes/` |
| Paper, official repo | `third_party/` |

Reproduction commands and known limits are in report Appendix B.

## Not in this repository

These are excluded by `.gitignore` and must be re-created locally:

| Path | How to re-create |
|---|---|
| `third_party/repo` | `git clone https://github.com/deepseek-ai/Engram third_party/repo` |
| `third_party/arxiv_src`, `third_party/pdf` | arXiv 2601.07372 v1/v2 PDFs and TeX source (`https://arxiv.org/e-print/2601.07372v2`) |
| `notes/external` | DeepSeek-V4.1-Flash `README.md`, `config.json`, `inference/*.py`, tech report (Hugging Face, revision `314113f8`) |
| `results/expA/corpus` | `scripts/expA_1_corpus.py` |
| `.venv` | Python 3.9 venv with `torch` (CPU), `transformers`, `tokenizers`, `datasets`, `numpy`, `sympy`, `matplotlib`, `pymupdf`, `opencc-python-reimplemented` |
