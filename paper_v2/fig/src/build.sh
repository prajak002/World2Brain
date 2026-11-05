#!/bin/sh
# Re-render the TikZ figures to PDF (run from paper_v2/): sh fig/src/build.sh
cd "$(dirname "$0")" && for f in pipeline multi controls; do tectonic -X compile $f.tex -o .. || pdflatex -output-directory=.. $f.tex; done
