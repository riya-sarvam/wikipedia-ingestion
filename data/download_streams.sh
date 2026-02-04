#!/usr/bin/env bash
#
# Download Wikipedia CirrusSearch content dump streams into data/streams/.
# Uses wget -c for resume. Default: enwiki_content, 20260201, files 00-62.
#
# Usage:
#   ./data/download_streams.sh
#   DATE=20250101 ./data/download_streams.sh
#   DATE=20260201 INDEX=enwiki_content ./data/download_streams.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STREAMS_DIR="${STREAMS_DIR:-$SCRIPT_DIR/streams}"
DATE="${DATE:-20260201}"
INDEX="${INDEX:-enwiki_content}"
# 00 to 62 (63 files) for enwiki_content; override COUNT=12 for a smaller test set
COUNT="${COUNT:-62}"

mkdir -p "$STREAMS_DIR"
cd "$STREAMS_DIR"

BASE="https://dumps.wikimedia.org/other/cirrus_search_index/${DATE}/index_name%3D${INDEX}"
echo "Downloading ${INDEX} ${DATE} (00000-000${COUNT}) into $STREAMS_DIR"

for i in $(seq -w 0 "$COUNT"); do
  file="${INDEX}-${DATE}-000${i}.json.bz2"
  url="${BASE}/${file}"
  wget -c "$url" -O "$file"
done

echo "Done. Files in $STREAMS_DIR"
