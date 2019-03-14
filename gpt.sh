#!/bin/bash
for p in $(ls ../S2_Products); do
	p3=${p:11:-46}
	sed -i "s/@@@@/$p/g" CNN_piloot_preprocessing.xml
	sed -i "s/TTTT/$p3/g" CNN_piloot_preprocessing.xml
	/home/paku/snap/bin/gpt CNN_piloot_preprocessing.xml -c 1G -q 3 -x -J-Xmx3G -J-Xms1G
	sed -i "s/$p/@@@@/g" CNN_piloot_preprocessing.xml
	sed -i "s/$p3/TTTT/g" CNN_piloot_preprocessing.xml
	break
done

#for f in $(ls ../Masks); do
#	index=$((${#f}-3))
#	f_extension="${f:$index:3}"
#	if [ "$f_extension" = "shp" ]; then
#		f2=${f::-4}
#		sed -i "s/@@@@/$f/g" CNN_piloot_preprocessing_2.xml
#		sed -i "s/%%%%/$f2/g" CNN_piloot_preprocessing_2.xml
#		f3="Mask_${f2}"
#		sed -i "s/TTTT/$f3/g" CNN_piloot_preprocessing_2.xml
#		/home/paku/snap/bin/gpt CNN_piloot_preprocessing_2.xml -c 1G -q 3 -x -J-Xmx3G -J-Xms1G
#		sed -i "s/$f/@@@@/g" CNN_piloot_preprocessing_2.xml
#		sed -i "s/$f3/TTTT/g" CNN_piloot_preprocessing_2.xml
#		sed -i "s/$f2/%%%%/g" CNN_piloot_preprocessing_2.xml
#	fi
#done
