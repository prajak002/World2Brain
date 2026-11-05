#!/bin/sh
# Build a clean Overleaf upload: sources + figures only, no macOS metadata, no build products.
# Usage (from repo root): sh paper_v2/scripts/make_overleaf_zip.sh
set -e
cd "$(dirname "$0")/.."
rm -f ../paper_v2.zip
COPYFILE_DISABLE=1 zip -X -r ../paper_v2.zip \
  main.tex main.bib cvpr.sty ieeenat_fullname.bst \
  fig/*.pdf fig/*.png fig/*.jpg fig/src README.md \
  -x '*.DS_Store' '__MACOSX/*'
unzip -l ../paper_v2.zip | tail -1
