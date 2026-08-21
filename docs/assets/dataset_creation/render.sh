#!/usr/bin/env bash
# Render every .mmd diagram in this folder to a .png next to it.
#
# Requires Node/npx. The first run downloads @mermaid-js/mermaid-cli and its
# bundled Chromium via puppeteer (a few hundred MB) — on a memory/disk
# constrained machine, run this manually and watch it rather than from CI.
#
# Usage: ./render.sh [scale]   (scale defaults to 2 for crisper PNGs)

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCALE="${1:-2}"

for mmd in "$DIR"/*.mmd; do
  name="$(basename "${mmd%.mmd}")"
  png="$DIR/${name}.png"
  echo "Rendering ${name}.mmd -> ${name}.png"
  npx --yes @mermaid-js/mermaid-cli \
    -i "$mmd" \
    -o "$png" \
    -b transparent \
    -s "$SCALE" \
    -p "$DIR/puppeteer-config.json"
done

echo "Done. PNGs written to $DIR"
