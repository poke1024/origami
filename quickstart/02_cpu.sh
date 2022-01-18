#!/bin/bash

conda activate origami_cpu

export ORIGAMI_WORK_DIR="/path/to/origami"  # !! adapt to your needs

# run.
export DATA_PATH="$ORIGAMI_WORK_DIR/data"

cd "$ORIGAMI_WORK_DIR/origami"
python -m origami.batch.detect.contours "$DATA_PATH"
python -m origami.batch.detect.flow "$DATA_PATH"
python -m origami.batch.detect.dewarp "$DATA_PATH"
python -m origami.batch.detect.layout "$DATA_PATH"
python -m origami.batch.detect.lines "$DATA_PATH"
python -m origami.batch.detect.order "$DATA_PATH"
