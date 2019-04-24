#!/bin/bash
GPT_PATH=/usr/local/snap6/bin/gpt
OUTPUT_PATH=/home/rus/shared/Bachelor_thesis/Scripts/
INPUT_DIM=/home/rus/shared/Bachelor_thesis/S2_Products/

for product in $(ls $INPUT_DIM); do
    prefix_product=${product:11:-46}
    sed "s@{{input_safe}}@${INPUT_DIM}${product}@g; s@{{prefix_dim}}@${OUTPUT_PATH}${prefix_product}@g" label_mask_preprocessing.in > label_mask_preprocessing_out.xml
    $GPT_PATH label_mask_preprocessing_out.xml -c 4G -q 3 -x -J-Xmx15G -J-Xms4G
    break
done

GPT_PATH=/usr/local/snap6/bin/gpt
MASK_PATH=/home/rus/shared/Bachelor_thesis/Masks/
OUTPUT_PATH=/home/rus/shared/Bachelor_thesis/Label_tifs/
INPUT_DIM=/home/rus/shared/Bachelor_thesis/Scripts/${prefix_product}.dim
for mask_file in $(ls $MASK_PATH); do
        index=$((${#mask_file}-3))
        mask_file_extension="${mask_file:$index:3}"
        if [ "${mask_file_extension}" = "shp" ]; then
                mask=${mask_file::-4}
		echo $mask
                prefix_mask="Mask_${mask}"
                sed "s@{{input_dim}}@${INPUT_DIM}@g; s@{{mask_file}}@${MASK_PATH}${mask_file}@g; s@{{mask}}@${mask}@g; s@{{prefix_mask}}@${OUTPUT_PATH}${prefix_mask}@g" label_mask_preprocessing_2.in > label_mask_preprocessing_out_2.xml
                $GPT_PATH label_mask_preprocessing_out_2.xml -c 4G -q 3 -x -J-Xmx15G -J-Xms4G
        fi
done

