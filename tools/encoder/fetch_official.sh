#!/usr/bin/env bash
set -euo pipefail
dest="${1:-third_party/llm2vec}"
revision=6bbd52528bee4936786ff0e9eb8a569698b1c731
if [[ ! -e "$dest" ]]; then
  git clone https://github.com/McGill-NLP/llm2vec.git "$dest"
  git -C "$dest" checkout --detach "$revision"
fi
[[ "$(git -C "$dest" rev-parse HEAD)" == "$revision" ]]
[[ -z "$(git -C "$dest" status --porcelain)" ]]
echo "Official scripts ready at $dest ($revision); no patches."
