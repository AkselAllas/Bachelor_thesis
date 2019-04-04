#!/bin/bash
GPT_PATH=/usr/local/snap6/bin/gpt
OUTPUT_PATH=/home/rus/shared/Bachelor_thesis/Scripts/
INPUT_DIM=/home/rus/shared/Bachelor_thesis/S2_Products/

for product in $(ls $INPUT_DIM); do
	prefix_product=${product:11:-46}
	sed "s@{{input_safe}}@${INPUT_DIM}${product}@g; s@{{prefix_dim}}@${OUTPUT_PATH}${prefix_product}@g" CNN_dim_mask_preprocessing.in > CNN_dim_mask_preprocessing_out.xml
	$GPT_PATH CNN_dim_mask_preprocessing_out.xml -c 4G -q 3 -x -J-Xmx15G -J-Xms4G
	break
done
