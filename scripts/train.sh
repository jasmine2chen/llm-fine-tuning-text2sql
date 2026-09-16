#!/bin/bash
set -euo pipefail
python -m text2sql.train --config configs/train.yaml "$@"
