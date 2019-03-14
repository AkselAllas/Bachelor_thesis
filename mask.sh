#!/bin/bash
for f in $(ls ../Masks); do
        index=$((${#f}-3))
        f_extension="${f:$index:3}"
        if [ "$f_extension" = "shp" ]; then
                f2=${f::-4}
                sed -i "s/@@@@/$f/g" CNN_piloot_preprocessing_2.xml
                sed -i "s/%%%%/$f2/g" CNN_piloot_preprocessing_2.xml
                f3="Mask_${f2}"
                sed -i "s/TTTT/$f3/g" CNN_piloot_preprocessing_2.xml
                /home/paku/snap/bin/gpt CNN_piloot_preprocessing_2.xml -c 1G -q 3 -x -J-Xmx3G -J-Xms1G
                sed -i "s/$f/@@@@/g" CNN_piloot_preprocessing_2.xml
                sed -i "s/$f3/TTTT/g" CNN_piloot_preprocessing_2.xml
                sed -i "s/$f2/%%%%/g" CNN_piloot_preprocessing_2.xml
        fi
done

