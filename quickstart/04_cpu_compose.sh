#!/bin/bash

conda activate origami_cpu

export ORIGAMI_WORK_DIR="/path/to/origami"  # !! adapt to your needs

# run.
export DATA_PATH="$ORIGAMI_WORK_DIR/data"

cd "$ORIGAMI_WORK_DIR/origami"
python -m origami.batch.detect.compose --page-xml "$DATA_PATH"
python -m origami.batch.utils.sample --all -a compose -o "$ORIGAMI_WORK_DIR/texts.zip" "$DATA_PATH"
