#!/bin/bash
TARGET_DIR=${1:-./waterbench_data}
DATA_PATH="$TARGET_DIR/data/OilSlick"

echo "Unpacking .tar files in $DATA_PATH..."

cd "$DATA_PATH" || exit
for f in *.tar; do 
    tar xf "$f"
    echo "Unpacked: $f"
done

echo "Unpacking completed! The images should now be in images_s1/."
