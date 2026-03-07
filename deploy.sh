#!/usr/bin/env bash
set -euo pipefail

DEST="nubo:~/static/puzzle-model/"

rsync -avzL --info=progress2 \
  explorer.html \
  explorer_data.json \
  data/raw/myspeedpuzzling/images \
  "$DEST"

echo "Deployed to $DEST"
