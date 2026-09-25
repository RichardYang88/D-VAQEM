#!/usr/bin/env bash
# Assemble a self-contained Advanced Quantum Technologies submission directory
# from the repository, ready to zip and upload to Wiley's submission system or
# to Overleaf.
#
#   bash paper/make_submission.sh [target-dir]
#
# The USG class, its style files, the bibliography styles and the STIX fonts
# live in LaTeX-template/ (the publisher's download); the manuscript, the
# generated tables, the bibliography and the figures live in the repository.
# This script copies both halves into one flat directory so that
# pdflatex/bibtex find everything without TEXINPUTS gymnastics.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TPL="$ROOT/LaTeX-template"
OUT="${1:-$ROOT/paper/AQT_submission}"

mkdir -p "$OUT/figures"

# --- class, styles, bibliography styles and fonts from the template --------
cp "$TPL/USG.cls" "$OUT/"
for f in "$TPL"/*.sty "$TPL"/*.STY "$TPL"/*.bst; do
    [ -e "$f" ] && cp "$f" "$OUT/"
done
cp -r "$TPL/Fonts" "$OUT/"
cp -r "$TPL/images" "$OUT/"

# --- manuscript sources ----------------------------------------------------
cp "$ROOT/paper/manuscript.tex" \
   "$ROOT/paper/tables.tex" \
   "$ROOT/paper/tables_supplement.tex" \
   "$ROOT/paper/references.bib" \
   "$ROOT/paper/toc_entry.tex" \
   "$OUT/"

# --- figures (the manuscript also finds them via ../figures/) --------------
cp "$ROOT"/figures/fig*.pdf "$OUT/figures/"

echo "submission tree assembled in $OUT"
echo "build with:"
echo "  cd $OUT && pdflatex manuscript && bibtex manuscript \\"
echo "             && pdflatex manuscript && pdflatex manuscript"
echo "the ToC entry (text + 55x50 mm graphic) is toc_entry.tex in the same tree"
