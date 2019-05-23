#!/bin/bash
GPT_PATH=/sar/aksel_snap/snap/bin/gpt
OUTPUT_PATH=/sar/aksel/Bachelor_thesis/Data/
INPUT_PATH=/sar/aksel/Bachelor_thesis/S2_Products/

for product in $(ls $INPUT_PATH); do
        index=$((${#product}-4))
        product_file_extension="${product:$index:4}"
        if [ "${product_file_extension}" = "SAFE" ]; then
                for (( k = 0; k < 2; k++ )); do
                        for (( l = 0; l < 2; l++ )); do
                                y=$( expr 5120 '*' "$l")
                                x=$( expr 5120 '*' "$k" + 740 )
                                subset_parameters="$x,$y,5120,5120"
                                prefix_product=${product:11:-46}
                                sed "s@{{input_safe}}@${INPUT_PATH}${product}@g; s@{{subset_params}}@${subset_parameters}@g; s@{{prefix_tif}}@${OUTPUT_PATH}${prefix_product}_${l}-${k}@g" bands_preprocessing.in > bands_preprocessing_out.xml
                                $GPT_PATH bands_preprocessing_out.xml -c 4G -q 3 -x -J-Xmx15G -J-Xms4G
                        done
                done
        fi
done
