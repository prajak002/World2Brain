# paper_v2: connectome-wired world models (reorganised draft)

Template: official CVPR author kit (`cvpr.sty`), `[review]` mode (anonymous, line numbers, confidential header); switch to `[pagenumbers]` for arXiv.
Figures: `scripts/make_figs.py` regenerates `fig/ladder.pdf`, `fig/curves.pdf`, `fig/realvid_bars.pdf` (numbers are a snapshot of `results/summary.md`); the strips, homing, activity and NYC frames are copied from `paper/fig` and `docs/media`.

Build: `tectonic main.tex` (or `pdflatex` + `bibtex` twice). Overleaf: upload `paper_v2.zip` (built by `scripts/make_overleaf_zip.sh`),
compiler pdfLaTeX, main document `main.tex`. The three TikZ diagrams (pipeline, multi-brain, controls) are pre-rendered to
`fig/*.pdf` so the free-plan compile stays well under the time limit; edit the sources in `fig/src/` and run
`sh fig/src/build.sh` to regenerate them. `main.pdf`, `main.blg` and `scripts/` are not part of the Overleaf upload.

Layout (2026-11-05): everything is in the single file `main.tex`, in CVPR order: preamble, generated numbers
(`\newcommand{\n...}` block from `scripts/make_paper_tables.py`, pasted from `paper/num.tex`), the `\TODO` placeholders
for run-2 numbers not yet produced (real video, transformer NYC homing), abstract, introduction, related work, method,
experiments (with the generated tables pasted in from `paper/tab/*.tex`, each marked `% ---- tab/<name>.tex ----`),
discussion, references, then the supplement as an appendix after `\clearpage`. Section banners `% ===== <name> =====`
mark where the old `sec/*.tex` files begin. To refresh numbers after a run: `PYTHONPATH=. python scripts/make_paper_tables.py`
in the repo root, then replace the corresponding blocks in `main.tex`.

Bib entries added in the 2026-11-04 literature pass are marked in `main.bib`; volumes/pages for entries flagged [M]
there were written from memory and should be spot-checked before submission.
