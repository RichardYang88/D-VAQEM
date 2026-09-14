# D-VAQEM paper — manuscript directory

Distribution-level variational quantum error mitigation (D-VAQEM) for learned
quantum metrology.  Target venue style: APS `revtex4-2` / `prresearch` (same
class as the companion VQ-CNNI manuscript in `../../paper/`).

## Contents

| file | what it is |
|---|---|
| `manuscript.tex` | the draft: abstract, 6 sections, 2 appendices (proofs, simulator validation), 6 figures, 3 main-text tables |
| `tables.tex` | **generated** main-text Tables I–III (`code/make_tables.py`) |
| `tables_supplement.tex` | **generated** Supplemental Tables S1–S9 (full per-setting matrices, E1/E3/E5 tables, seed and ablation tables) |
| `references.bib` | bibliography (companion-paper entries + QEM / metrology / statistics entries) |
| `check_tex.py` | integrity checker: environment balance, `\ref`/`\label`, `\cite` vs bib keys, truncation heuristic |
| `../figures/` | **generated** figures (PDF for the manuscript, PNG for inspection) |

## Building

The manuscript needs a TeX distribution with `revtex4-2` (TeX Live
`texlive-publishers` + `texlive-science`, or MikTeX) and `bibtex`:

```bash
cd new_paper/paper
pdflatex manuscript && bibtex manuscript && pdflatex manuscript && pdflatex manuscript
```

`manuscript.tex` inputs `tables.tex`, `tables_supplement.tex` and the figures
from `../figures/` via `\graphicspath{{../figures/}}`, so build from this
directory.  No TeX distribution is installed on the analysis machine that
produced this draft; `check_tex.py` was used instead to verify environment
balance, label/reference resolution, citation keys and figure paths.

## Regenerating everything from the results

```bash
cd ../code
python collect_numbers.py --tag final     # results/paper_numbers.{md,json}
python make_tables.py     --tag final     # paper/tables*.tex
python make_figures.py    --tag final     # ../figures/fig*.{pdf,png}
python check_tex.py                       # manuscript integrity
```

Every number in `manuscript.tex` is taken from
`../results/paper_numbers.md` (machine-readable mirror:
`../results/paper_numbers.json`); the tables and figures are generated
directly from `../results/*.json` / `*.npz`, so text, tables and figures cannot
drift apart.  See `../README.md` for how the results themselves were produced.

## Status / TODO before submission

* Author list and affiliations are carried over from the companion VQ-CNNI
  manuscript; confirm before submission.
* `yang2025vqcnni` in `references.bib` is a placeholder for the companion
  manuscript (update journal/volume once published).
* Compile once with a full TeX installation and fix any overfull boxes in the
  wide Supplemental tables (they are set in `\tiny` in `table*`).
