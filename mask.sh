#!/bin/bash
GPT_PATH=/usr/local/snap6/bin/gpt
MASK_PATH=/home/rus/Aksel/Andmed/Masks/
OUTPUT_PATH=/home/rus/Aksel/Andmed/
INPUT_DIM=/home/rus/Aksel/Andmed/20180520.dim

for mask_file in $(ls ../../Andmed/Masks); do
        index=$((${#mask_file}-3))
        mask_file_extension="${mask_file:$index:3}"
        if [ "${mask_file_extension}" = "shp" ]; then
                mask=${mask_file::-4}
		echo $mask
                prefix_mask="Mask_${mask}"
                sed "s@{{input_dim}}@${INPUT_DIM}@g; s@{{mask_file}}@${MASK_PATH}${mask_file}@g; s@{{mask}}@${mask}@g; s@{{prefix_mask}}@${OUTPUT_PATH}${prefix_mask}@g" CNN_piloot_preprocessing.in > CNN_piloot_preprocessing_out.xml
                $GPT_PATH CNN_piloot_preprocessing_out.xml -c 4G -q 3 -x -J-Xmx15G -J-Xms4G
        fi
done

