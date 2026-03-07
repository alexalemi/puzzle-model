#!/usr/bin/env bash
set -euo pipefail

DEST="nubo:~/static/puzzle-model/"

rsync -avzL --info=progress2 \
  explorer.html \
  explorer_data.json \
  explorer_puzzler_obs.json \
  "$DEST"

rsync -avzL --info=progress2 \
  data/raw/myspeedpuzzling/images \
  "$DEST"data/raw/myspeedpuzzling/

echo "Deployed to $DEST"
