#!/bin/bash
for p in $(ls ../S2_Products); do
	p2=${p::-5}
	p3=${p:11:-46}
	sed -i "s/@@@@/$p/g" CNN_piloot_preprocessing.xml
	/usr/local/snap6/bin/gpt CNN_piloot_preprocessing.xml -c 3G -q 4 -x -J-Xmx12G -J-Xms1G
	sed -i "s/$p/@@@@/g" CNN_piloot_preprocessing.xml
	for f in $(ls ../Masks); do
		index=$((${#f}-3))
		f_extension="${f:$index:3}"
		if [ "$f_extension" = "shp" ]; then
			sed -i "s/@@@@/$f/g" CNN_piloot_preprocessing_2.xml
			f2=${f::-4}
			a="::Subset_${p2}_resampled"
			geo="$f2$a"
			sed -i "s/%%%%/$f2/g" CNN_piloot_preprocessing_2.xml
			f3="${p3}_${f2}"
			sed -i "s/TTTT/$f3/g" CNN_piloot_preprocessing_2.xml
			/usr/local/snap6/bin/gpt CNN_piloot_preprocessing_2.xml -c 3G -q 4 -x -J-Xmx12G -J-Xms1G
			sed -i "s/$f/@@@@/g" CNN_piloot_preprocessing_2.xml
			sed -i "s/$f3/TTTT/g" CNN_piloot_preprocessing_2.xml
			sed -i "s/$f2/%%%%/g" CNN_piloot_preprocessing_2.xml
		fi
	done
done

