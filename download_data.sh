#!/bin/bash
TARGET_DIR=${1:-./waterbench_data}

echo "Starting download of the WaterBench OilSlick dataset to $TARGET_DIR..."

hf download ayushprd/WaterBench --repo-type dataset \
--include 'data/OilSlick/OilSlick-images_s1-00.tar' \
'data/OilSlick/OilSlick-images_s1-01.tar' \
'data/OilSlick/metadata.csv' \
'data/OilSlick/metadata.json' \
'data/OilSlick/splits/random/*.txt' \
'data/OilSlick/splits/geographic/*.txt' \
--local-dir "$TARGET_DIR"

echo "Download finished."
