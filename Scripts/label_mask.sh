#!/bin/bash
echo "================ Phase 1 ================="

GPT_PATH=/sar/aksel_snap/snap/bin/gpt
OUTPUT_PATH=/sar/aksel_snap/
INPUT_PATH=/home/aksel/Bachelor_thesis/S2_Products/

#Generate .dim files, which are needed to make label masks 
for product in $(ls $INPUT_PATH); do
        index=$((${#product}-4))
        product_file_extension="${product:$index:4}"
        prefix_product=${product:11:-46}
        #break #Uncomment this to skip Phase 1
        if [ "${product_file_extension}" = "SAFE" ]; then
                for (( k = 0; k < 2; k++ )); do
                        for (( l = 0; l < 2; l++ )); do
                                y=$( expr 5120 '*' "$k")
                                x=$( expr 5120 '*' "$l" + 740 )
                                subset_parameters="$x,$y,5120,5120"
                                sed "s@{{input_safe}}@${INPUT_PATH}${product}@g; s@{{subset_params}}@${subset_parameters}@g; s@{{prefix_dim}}@${OUTPUT_PATH}${prefix_product}_${l}-${k}@g" label_mask_preprocessing.in > label_mask_preprocessing_out.xml
                                $GPT_PATH label_mask_preprocessing_out.xml -c 4G -q 3 -x -J-Xmx15G -J-Xms4G
                        done
                done
		break
        fi
done

echo "================ Phase 2 ================="
#Make label masks
#This script throws an error, when there are no fields in the current tile. That means no file will be made for that class/tile combination. We will generate empty files in Phase 3

MASK_PATH=/home/aksel/Bachelor_thesis/Masks/
OUTPUT_PATH=/sar/aksel_snap/Label_tifs/
for (( k = 0; k < 2; k++ )); do
        for (( l = 0; l < 2; l++ )); do
                #break #Uncomment this to skip Phase 2
                INPUT_PATH=/sar/aksel_snap/${prefix_product}_${l}-${k}.dim
                for mask_file in $(ls $MASK_PATH); do
                        index=$((${#mask_file}-3))
                        mask_file_extension="${mask_file:$index:3}"
                        if [ "${mask_file_extension}" = "shp" ]; then
                                mask=${mask_file::-4}
                                echo $mask
                                prefix_mask="Mask_${mask}"
                                sed "s@{{input_dim}}@${INPUT_PATH}@g; s@{{mask_file}}@${MASK_PATH}${mask_file}@g; s@{{mask}}@${mask}@g; s@{{prefix_mask}}@${OUTPUT_PATH}${prefix_mask}_${l}-${k}@g" label_mask_preprocessing_2.in > label_mask_preprocessing_out_2.xml
                                $GPT_PATH label_mask_preprocessing_out_2.xml -c 4G -q 3 -x -J-Xmx15G -J-Xms4G 2>>error_tiles.txt 
                                echo "${mask}_${l}-${k}">>error_tiles.txt
                        fi
                done
        done
done

echo "================ Phase 3 ================="
#Generate empty files, so that during training, the amount of classes would still be the same for every tile.

for missing_file in `sed -n '/Error/{n;p}' error_tiles.txt`; do
    touch /sar/aksel_snap/Label_tifs/Mask_${missing_file}.tif
done
