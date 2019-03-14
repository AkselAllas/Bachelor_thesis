#!/bin/bash
for p in $(ls ../S2_Products); do
	p3=${p:11:-46}
	sed -i "s/@@@@/$p/g" CNN_piloot_preprocessing.xml
	sed -i "s/TTTT/$p3/g" CNN_piloot_preprocessing.xml
	/home/paku/snap/bin/gpt CNN_piloot_preprocessing.xml -c 1200M -q 3 -x -J-Xmx3G -J-Xms1G
	sed -i "s/$p/@@@@/g" CNN_piloot_preprocessing.xml
	sed -i "s/$p3/TTTT/g" CNN_piloot_preprocessing.xml
	break
done
