"""check_tex.py -- lightweight integrity check for the manuscript."""
import re
import sys

tex = open(sys.argv[1] if len(sys.argv) > 1 else "manuscript.tex").read()
for f in ("tables.tex", "tables_supplement.tex"):
    try:
        tex += "\n" + open(f).read()
    except OSError:
        pass
for env in sorted(set(re.findall(r"\\begin\{(\w+\*?)\}", tex))):
    b = len(re.findall(r"\\begin\{%s\}" % env, tex))
    e = len(re.findall(r"\\end\{%s\}" % env, tex))
    if b != e:
        print("UNBALANCED", env, b, e)
print("figure labels:", re.findall(r"\\label\{(fig:\w+)\}", tex))
# heuristic: a paragraph that ends mid-sentence (truncation detector)
body = tex.split("\\appendix")[0]
bad = []
for blk in body.split("\n\n"):
    lines = [l for l in blk.strip().split("\n") if l.strip()]
    if not lines:
        continue
    last = lines[-1].strip()
    if last.startswith("\\") and not last.endswith("$"):
        continue
    if last.endswith(("\\begin{equation}", "\\begin{figure*}")):
        continue
    if not re.search(r"[.:;%)$\]]$|\\\\$", last):
        bad.append(last[-70:])
print("suspect paragraph endings:")
for b in bad:
    print("   ...", b)
print("lines:", tex.count("\n"))

