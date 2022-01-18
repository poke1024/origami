#!/bin/bash

export ORIGAMI_WORK_DIR="/path/to/origami"  # !! adapt to your needs

conda activate origami_gpu

# run.
export DATA_PATH="$ORIGAMI_WORK_DIR/data"
export MODEL_PATH="$ORIGAMI_WORK_DIR/models"

cd "$ORIGAMI_WORK_DIR/origami"
python -m origami.batch.detect.segment "$DATA_PATH" --model "$MODEL_PATH"
