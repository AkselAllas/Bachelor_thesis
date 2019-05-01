#!/bin/bash
GPT_PATH=/opt/snap6/bin/gpt
OUTPUT_PATH=/home/aksel/Bachelor_thesis/Data/
INPUT_PATH=/home/aksel/Bachelor_thesis/S2_Products/

for product in $(ls $INPUT_PATH); do
        prefix_product=${product:11:-46}
        sed "s@{{input_safe}}@${INPUT_PATH}${product}@g; s@{{prefix_tif}}@${OUTPUT_PATH}${prefix_product}@g" bands_preprocessing.in > bands_preprocessing_out.xml
        $GPT_PATH bands_preprocessing_out.xml -c 4G -q 3 -x -J-Xmx15G -J-Xms4G
done
