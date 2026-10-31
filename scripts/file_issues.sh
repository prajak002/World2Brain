#!/bin/zsh
# File every .github/issues/*.md as a GitHub issue (front-matter: title, labels). Creates labels if missing.
cd "$(dirname "$0")/.." || exit 1
for lbl in P0 P1 P2 controls experiment model baselines data teacher evaluation neuroscience statistics planning benchmark performance writing; do
  gh label create "$lbl" --force >/dev/null 2>&1
done
for f in .github/issues/*.md; do
  title=$(sed -n 's/^title: "\(.*\)"$/\1/p' "$f")
  labels=$(sed -n 's/^labels: //p' "$f" | tr -d ' ')
  body=$(awk 'BEGIN{n=0} /^---$/{n++; next} n>=2' "$f")
  gh issue create --title "$title" --label "$labels" --body "$body"
done
