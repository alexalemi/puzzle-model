#!/usr/bin/env bash
set -euo pipefail

DEST="nubo:~/static/puzzle-model/"

rsync -avz --delete --info=progress2 \
  explorer.html \
  explorer_data.json \
  puzzle_images \
  "$DEST"

echo "Deployed to $DEST"
