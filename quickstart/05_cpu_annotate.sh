#!/bin/bash

conda activate origami_cpu

export ORIGAMI_WORK_DIR="/path/to/origami"  # !! adapt to your needs

# run.
export DATA_PATH="$ORIGAMI_WORK_DIR/data"

cd "$ORIGAMI_WORK_DIR/origami"

python -m origami.batch.annotate.contours "$DATA_PATH"
python -m origami.batch.annotate.flow "$DATA_PATH"
python -m origami.batch.annotate.grid "$DATA_PATH"
python -m origami.batch.annotate.layout "$DATA_PATH"
python -m origami.batch.annotate.lines "$DATA_PATH"
