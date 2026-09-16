#!/bin/bash
set -euo pipefail
python -m text2sql.evaluate --model-path ./checkpoints/text2sql --config configs/eval.yaml "$@"
