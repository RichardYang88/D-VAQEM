# D-VAQEM paper — manuscript directory

Distribution-level variational quantum error mitigation (D-VAQEM) for learned
quantum metrology.  Target venue style: APS `revtex4-2` / `prresearch` (same
class as the companion VQ-CNNI manuscript in `../../VQ-CNNI/paper/`).

## Contents

| file | what it is |
|---|---|
| `manuscript.tex` | the draft: abstract, 6 sections, 2 appendices (proofs, simulator validation), 6 figures, 4 main-text tables |
| `tables.tex` | **generated** main-text Tables I–III (`code/make_tables.py`) |
| `tables_supplement.tex` | **generated** Supplemental Tables S1–S9 (full per-setting matrices, E1/E3/E5 tables, seed and ablation tables) |
| `references.bib` | bibliography (companion-paper entries + QEM / metrology / statistics entries) |
| `check_tex.py` | integrity checker: environment balance, `\ref`/`\label`, `\cite` vs bib keys, truncation heuristic |
| `../figures/` | **generated** figures (PDF for the manuscript, PNG for inspection) |

## Building

The manuscript needs a TeX distribution with `revtex4-2` (TeX Live
`texlive-publishers` + `texlive-science`, or MikTeX) and `bibtex`:

```bash
cd paper        # the repository root is D-VAQEM/
pdflatex manuscript && bibtex manuscript && pdflatex manuscript && pdflatex manuscript
```

`manuscript.tex` inputs `tables.tex` and `tables_supplement.tex` and locates the
figures through `\graphicspath{{./}{../figures/}{figures/}}`, so it builds both
in the repository layout and from a flat Overleaf upload (`manuscript.tex` +
`tables*.tex` + `references.bib` + `fig*.pdf` in one directory).  The generated
tables use the `\fitwidth` macro defined in the manuscript preamble, so they
must be compiled as part of `manuscript.tex`.

No TeX distribution is installed on the analysis machine that produced this
draft; `check_tex.py` was used instead to verify environment balance,
label/reference resolution, citation keys and figure paths.

## Layout conventions (after the first Overleaf compile)

* **Figures are declared where they are first cited.**  All six `figure*`
  environments used to sit in a single block after the Conclusion, so LaTeX ran
  out of room for wide floats and flushed them at the end of the document —
  which is why the figures appeared interleaved with the reference list.  Each
  float now sits in the paragraph that first cites it, and the preamble relaxes
  `\topfraction`/`\dbltopfraction` and raises `topnumber`/`totalnumber`.
* **`\clearpage` around the bibliography.**  No float can be carried into the
  reference pages and the Supplemental tables start on a fresh page.  The
  bibliography is typeset in the same single-column grid as the rest of the
  preprint (the previous `\twocolumngrid` switch squeezed 12-author entries
  into a narrow column), with an explicit `\bibliographystyle{apsrev4-2}`.
* **No table can be wider than the page.**  `make_tables.py` wraps every tabular
  in `\fitwidth`, which applies `\resizebox` *only* when the natural width
  exceeds `\textwidth`.  Table II uses a grouped two-row header
  (`depol. | deph. | amp. damp. | readout` over `∞ | 1024`) instead of eight
  long column names, and Supplemental Tables S1/S2 rotate their 16 setting
  names so they can be set in `\scriptsize` rather than `\tiny`.  Measured
  natural widths (Computer Modern metrics, `\textwidth` = 469 pt): Table II
  58 % (was 117 %), S1/S2 86 % (was 131 %), every other table ≤ 96 %.  The
  numeric content of all 138 data rows is unchanged by this reformatting.
* **`references.bib` is normalized.**  Journal names use the standard APS
  abbreviations, the three arXiv entries read `journal = {arXiv:NNNN.NNNNN}`,
  the IEEE QCE paper is `@inproceedings` (it was `@article` with a proceedings
  name in the `journal` field) and the companion manuscript is `@misc` with a
  `note` (it used to print `companion manuscript` as a journal name).

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
* Compile once with a full TeX installation (Overleaf is fine) and confirm the
  page breaks.  The table widths are checked numerically here (`\fitwidth`,
  compact/rotated headers), but no TeX engine is installed on the analysis
  machine, so the final overfull-box check has to happen on Overleaf.
* `\preprint{APS/123-QED}` is still a placeholder; remove it for submission.
* The code-availability statement names the repository but carries no public
  URL yet; insert one when D-VAQEM is made public.
