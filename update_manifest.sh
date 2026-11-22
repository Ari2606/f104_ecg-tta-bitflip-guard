#!/bin/bash
# Refresh the hash manifest and make sure the old project name does not appear anywhere.
cd "$(dirname "$0")"
if grep -rIil "physioguard" . --exclude-dir=.git --exclude=update_manifest.sh; then
  echo "Old project name found in the files above. Rename it before committing."; exit 1
fi
find . -type f -not -path './.git/*' -not -name MANIFEST.sha256 | sort | xargs sha256sum > MANIFEST.sha256
echo "manifest updated: $(wc -l < MANIFEST.sha256) files"
