#!/bin/bash

export ORIGAMI_WORK_DIR="/path/to/origami"  # !! adapt to your needs

conda activate origami_gpu

export DATA_PATH="$ORIGAMI_WORK_DIR/data"
export MODEL_PATH="$ORIGAMI_WORK_DIR/origami-v19-h56-styled-v2"

cd "$ORIGAMI_WORK_DIR/origami"
python -m origami.batch.detect.ocr "$DATA_PATH" --model "$MODEL_PATH" -b 8
