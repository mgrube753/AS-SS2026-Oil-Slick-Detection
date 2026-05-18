#!/bin/bash

TARGET_DIR=${1:-./10_waterbench_data}
DATA_PATH="$TARGET_DIR/data/OilSlick"

if [ -d "$DATA_PATH" ]; then
    read -p "Data already exists in $TARGET_DIR. Overwrite to repair? (y/n): " answer
    if [[ "$answer" != "y" ]]; then
        echo "Aborted."
        exit 0
    fi
fi

# ========================

echo "Starting download of the WaterBench OilSlick dataset to $TARGET_DIR..."

hf download ayushprd/WaterBench --repo-type dataset \
'data/OilSlick/OilSlick-images_s1-00.tar' \
'data/OilSlick/OilSlick-images_s1-01.tar' \
'data/OilSlick/metadata.csv' \
'data/OilSlick/metadata.json' \
'data/OilSlick/splits/random/*.txt' \
'data/OilSlick/splits/geographic/*.txt' \
--local-dir "$TARGET_DIR"

echo "Download finished."

# ========================

echo "Unpacking .tar files in $DATA_PATH..."

cd "$DATA_PATH" || exit
for f in *.tar; do 
    tar xf "$f"
    echo "Unpacked: $f"
done

echo "Unpacking completed! The images should now be in images_s1/."