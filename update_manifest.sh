#!/bin/bash
# Refresh the hash manifest and make sure the old project name does not appear anywhere.
cd "$(dirname "$0")"
PAT="physio""guard"
if grep -rIil "$PAT" . --exclude-dir=.git; then
  echo "Old project name found in the files above. Rename it before committing."; exit 1
fi
find . -type f -not -path './.git/*' -not -name MANIFEST.sha256 | sort | xargs sha256sum > MANIFEST.sha256
echo "manifest updated: $(wc -l < MANIFEST.sha256) files"
