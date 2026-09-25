# D-VAQEM paper — manuscript directory

Distribution-level variational quantum error mitigation (D-VAQEM) for learned
quantum metrology.  **Target venue: *Advanced Quantum Technologies*
(Wiley-VCH)**, formatted with the publisher's `USG` class and the `ASNA` option
(Chicago numbered references), i.e. the two-column layout and the back-matter
order required by the journal's *Manuscript Preparation Checklist*.  The earlier
APS `revtex4-2`/`prresearch` formatting has been superseded; the publisher's
class and style files live in `../LaTeX-template/`.

## Contents

| file | what it is |
|---|---|
| `manuscript.tex` | the submission file: Wiley front matter (title, authors, affiliations, corresponding-author and funding blocks, abstract, keywords), 6 sections, 4 appendices (proofs, simulator validation, ablation, ZNE in the blind limit), 6 figures, Tables I–IV |
| `tables.tex` | **generated** main-text Tables I–IV (`code/make_tables.py`), three-line booktabs rules |
| `tables_supplement.tex` | **generated** Supplemental Tables S1–S13 (per-setting matrices, sufficiency/scaling diagnostics, seed and ablation repeats, paired bootstrap intervals, per-setting ZNE comparison, claim-by-claim verdict table) |
| `toc_entry.tex` | the journal's Table-of-Contents entry: a 58-word text plus the 55 mm × 50 mm graphic (`../figures/fig_toc.pdf`); compile separately and upload the one-page PDF |
| `references.bib` | bibliography (companion-paper entries + QEM / metrology / statistics entries); the manuscript gives no `\bibliographystyle` because the class supplies the Chicago numbered style |
| `check_tex.py` | integrity checker: environment balance, `\ref`/`\label`, `\cite` vs bib keys, truncation heuristic, **plus** a cross-check that every statistical figure and every wall-clock cost quoted in the prose is re-derived from `../results/*.json` (exits non-zero on a mismatch) |
| `make_submission.sh` | assembles a flat, self-contained submission tree (class + styles + fonts + manuscript + tables + figures) in `AQT_submission/` |
| `../figures/` | **generated** figures (PDF for the manuscript, PNG for inspection, `fig_toc` for the ToC entry) |

## Building

No TeX distribution is installed on the analysis machine, so the manuscript is
validated here by `check_tex.py` and compiled on Overleaf.  The `USG` class needs
the publisher's style files and the STIX fonts that ship with it, so assemble
everything into one directory first:

```bash
bash paper/make_submission.sh          # -> paper/AQT_submission/
cd paper/AQT_submission
pdflatex manuscript && bibtex manuscript && pdflatex manuscript && pdflatex manuscript
pdflatex toc_entry                     # the Table-of-Contents page
```

`../LaTeX-template/` is **not tracked by git** — it is the publisher's 24 MB
download (`USG.cls`, the style and `.bst` files, `Fonts/`, `images/`, the sample
`Optimal-Design-layout.tex` and the manuscript-preparation checklist), so unpack
the journal's Wiley LaTeX zip there before running the script.
`make_submission.sh` copies the class, every `.sty`/`.STY`/`.bst`, `Fonts/` and
`images/` into the submission tree together with `manuscript.tex`,
`tables*.tex`, `references.bib`, `toc_entry.tex` and `../figures/fig*.pdf`; the
generated `paper/AQT_submission/` is ignored as well and can be rebuilt at will.

`manuscript.tex` inputs `tables.tex` and `tables_supplement.tex` and locates the
figures through `\graphicspath{{./}{../figures/}{figures/}}`, so it builds both
in the repository layout and from the flat submission tree.  The generated
tables use the `\fitwidth` macro defined in the manuscript preamble, so they
must be compiled as part of `manuscript.tex`.

## Formatting conventions required by the journal

* **Three-line tables.**  `make_tables.py` emits `\toprule/\midrule/\bottomrule`
  (the template loads `booktabs`); the checklist asks for horizontal lines only,
  and the class prints captions itself in small roman type with a bold
  "**Table n.**" label, so there is no `\hline`, no vertical rule and no manual
  caption styling.  `\tnote`/`tablenotes` do not exist in `USG.cls`, so table
  notes are plain sentences under the table and dagger markers became `a`/`b`/`c`.
* **Wide floats span both columns.**  Tables II and III, the Supplemental tables
  S1–S5, S9, S10, S12 and S13, and all six figures use `table*`/`figure*` at `width=\linewidth`
  (17.5 cm of text width); `\fitwidth` resizes a `tabular` *only* when its
  natural width exceeds `\linewidth`, so nothing overflows the column.  Table II
  keeps its grouped two-row header (`depol. | deph. | amp. damp. | readout` over
  `inf | 1024`) and S1/S2 keep their rotated setting names.  The numeric content
  of all data rows is unchanged by this reformatting.
* **Units in square brackets.**  Axis labels, table headers and prose read
  `[dB]`, `[shots]`, `[s]` rather than `(dB)`, `(s)`.
* **US spelling** throughout the body text, the generated tables and the
  generated figure labels: `analyze`, `optimize`, `depolarizing`,
  `regularization`, `characterization`, `utilize`, `behavior`, `color`,
  `modeling`, …  (`check_tex.py` re-derives every quoted number from the result
  files, so the prose cannot drift while it is edited.)
* **Back matter in the checklist's order**: Author Contributions →
  Acknowledgments → Conflicts of Interest → Data Availability Statement →
  references → Supporting Information, each a `\bmsubsection*`, word-for-word
  the sequence used by the publisher's sample `Optimal-Design-layout.tex`.
* **Front matter order and limits.**  The checklist fixes the file order as
  Title – Authors – Affiliations – Keywords – Abstract – Main Text, caps the
  abstract at 200 words (190 now) and the keyword list at ten entries (six now,
  pipe-separated exactly as in the publisher's sample).
* **Panels are labeled `(a)`, `(b)`, …** in lower-case bold type, drawn by the
  single `panel()` helper in `make_figures.py`.  Legends stay inside the floats
  instead of being collected at the end of the file, which is what the
  publisher's own LaTeX sample does (that instruction is for the Word template).
* **Table notes live in the caption.**  The checklist wants lowercase letters
  with a closing parenthesis for table footnotes; the generated tables have no
  footnote rows at all — every qualifier is a sentence inside the caption — so
  there is nothing to renumber.
* **Citations are inline square brackets** numbered in order of citation: the
  `ASNA` option loads `NJDnatbib` with `\setcitestyle{numbers,square}` and
  selects `wileyNJD-Chicago.bst`, so `\cite` needs no manual formatting and the
  manuscript sets no `\bibliographystyle`.
* **Appendices stay in the article file.**  The four appendices (proofs,
  simulator validation, ablation, ZNE in the blind limit) are typeset in the
  same two-column grid as the body — `USG.cls` offers no `\onecolumngrid`, and
  every display equation is short enough for one column — and they precede the
  back matter, so the checklist's required blocks stay contiguous at the end of
  the file and the Supplemental tables follow the Supporting Information note.
* **Figures are declared where they are first cited** and the preamble relaxes
  `\topfraction`/`\dbltopfraction` and raises `topnumber`/`totalnumber`, so wide
  floats are not flushed to the end of the document; `\clearpage` before the
  bibliography keeps floats out of the reference pages.
* **`references.bib` is normalized.**  Journal names use the standard
  abbreviations, the three arXiv entries read `journal = {arXiv:NNNN.NNNNN}`,
  the IEEE QCE paper is `@inproceedings` (it was `@article` with a proceedings
  name in the `journal` field) and the companion manuscript is `@misc` with a
  `note` (it used to print `companion manuscript` as a journal name).

## Regenerating everything from the results

```bash
cd ../code
python collect_numbers.py --tag final     # results/paper_numbers.{md,json}
                                          #   (+ results/ci_final.json, see below)
python make_tables.py     --tag final     # paper/tables*.tex
python make_figures.py    --tag final     # ../figures/fig*.{pdf,png} + fig_toc
cd ../paper && python check_tex.py        # manuscript integrity + number audit
```

`collect_numbers.py` calls `bootstrap_ci.build()` on the headline sweep and writes
`../results/ci_final.json`, the paired bootstrap intervals and exact tests behind
Supplemental Tables S11–S13; `bootstrap_ci.py` can also be run standalone
(`python bootstrap_ci.py --tag final`) and prints the same digest.  It needs the
per-phase / per-trial arrays that `run_experiments.metrics(detail=True)` stores,
so a sweep produced before that change yields no intervals (the digest says so
explicitly rather than failing silently).

`collect_numbers.py` additionally calls `bootstrap_ci.build_pec()` on
`../results/e6_pec.json` and writes `../results/ci_pec_final.json`, the intervals
and exact tests behind the probabilistic-error-cancellation subsection
(Sec. `subsec:pec`).  E6 is archived separately from the sweep because it is
evaluated on its own 20-setting grid and its purpose is to price the
quasi-probability route rather than to rank methods; `build_pec` nonetheless
reuses the *same* resampling primitives, so a PEC interval means exactly what a
D-VAQEM interval means.  Pass `--no-pec` to `bootstrap_ci.py` to skip that block.

Every number in `manuscript.tex` is taken from
`../results/paper_numbers.md` (machine-readable mirror:
`../results/paper_numbers.json`); the tables and figures are generated
directly from `../results/*.json` / `*.npz`, so text, tables and figures cannot
drift apart.  See `../README.md` for how the results themselves were produced.

## Verification status

All gates are green on this machine: `check_tex.py` (66 quoted numbers agree, 0
mismatches, no undefined references), `code/test_bootstrap_ci.py` (78 checks),
`code/test_pec.py` (57 checks), `code/validate_simulator.py` (53 checks,
including the independent PennyLane density-matrix comparison) and
`code/smoke_test.py`.  Every front- and back-matter macro used by
`manuscript.tex` (`\articletype`, `\journal`, `\volume`, `\copyyear`,
`\startpage`, `\articledoi`, `\titlemark`, `\authormark`, `\address`,
`\orgdiv`/`\orgname`/`\orgaddress`/`\state`/`\country`, `\corres`,
`\fundingInfo`, `\keywords`, `\bmsubsection`) is defined by `USG.cls`, and no
`revtex4-2` macro survives in the file.  The overfull-box check still has to
happen on Overleaf.

American spelling is enforced in every *paper-facing* file — `manuscript.tex`,
the generated `tables*.tex`, the generated figure labels and `toc_entry.tex` —
so `artefact`, `centred`, `favourable`, `Cancelling`, `towards` and
`depolarising` no longer appear in anything a reader sees.  They do survive
inside `collect_numbers.py` / `bootstrap_ci.py` (the `least_favourable_*` JSON
keys and the prose of the archived `results/paper_numbers.md` digest), which are
deliberately left untouched so the archived artifacts stay byte-identical to the
digest that `check_tex.py` audits.  Panel labels, bracketed units and the ToC
page size are generated, not hand-set.

## Status / TODO before submission

* **Author metadata**: add each author's ORCID ID and confirm the initials, the
  author order and the affiliation split (search `TODO(author)` in
  `manuscript.tex`).  The front matter uses *Qingchuan Yang*, *Xianing Feng* and
  *Lianfu Wei* with Lianfu Wei as corresponding author, as in the companion
  VQ-CNNI manuscript; the Author Contributions block now carries the matching
  initials (the earlier draft had stale ones).
* **Funding text**: `\fundingInfo` carries NKRDC Grant No. 2021YFA0718803 and a
  generic NSFC sentence; fill in the NSFC grant number(s) and any additional
  funder, and mirror the same wording in the Acknowledgments.
* **Data availability**: the statement points at the public repository and
  describes its contents; mint the Zenodo archive on acceptance and insert the
  DOI (no placeholder DOI is printed in the current text).
* `yang2025vqcnni` in `references.bib` is a placeholder for the companion
  manuscript (update journal/volume once published).
* **Compile on Overleaf** from the `make_submission.sh` tree and check the page
  breaks, the two-column float placement and any overfull boxes.
* **Table-of-Contents entry**: compile `toc_entry.tex` and upload the one-page
  PDF.  `figures/fig_toc.pdf` is the one figure saved without the tight crop, so
  its page measures exactly 55.00 mm × 50.00 mm (`pdfinfo`), and the text is 58
  words against the required 50–60.
* **Supporting Information as its own file**: the checklist asks for the
  supporting text and graphics in a single separate file, so at submission the
  Supplemental tables (S1–S13, currently appended to `manuscript.tex` after the
  Supporting Information note so that reviewers see one document) should be
  split out into their own SI PDF.
* **Author contributions**: the current split (conception by Q. Y. and L. W.,
  implementation and analysis by Q. Y., writing by all authors) is plausible but
  must be confirmed, and X. F.'s specific role should be stated.
