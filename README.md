# Bakalaureuse töö - Aksel Allas

## Kaustade struktuur:

##### Juurkaust ./ 

Sisaldab masinõppeks vajalikke python OOP klasse: ./s2_preprocessor.py, ./s2_model.py, ./data_generator.py ning graafikute kuvamise klassi ./plotter.py.

Sisaldab fit() funktsiooni kasutavat masinõppe treenimise skripti ./fit_main.py.

Sisaldab fit_generator() funktsiooni kasutavat masinõppe treenimise skripti ./generator_main.py.

Sisaldab ennustuste tegemise skripti ./predict.py.

Sisaldab klasside kaalude .npy faile.

Sisaldab funktsioone uurimisalasse kuuluvate töötletud ruutude listi tekitamiseks ./read_processed_tiles.py ning uurimisalasse kuuluvate ruutude objekte processed_tile_dict.npy ja zero_tile_dict.npy 

##### S2_products

Siia tuleb panna Sentinel-2 pildid .SAFE formaadis.

##### Masks 

Siia tuleb panna klassifitseeritavata klasside põldude vektorkihid .shp formaadis.

##### ROI 

Siia tuleb panna uuritava huviala vektorkiht .shp formaadis. NB! kõikidel .shp failidel peab olema sama koordinaatsüsteem.

##### Gpt_scripts

Sisaldab bash skripte, mis kasutavad omakorda SNAP gpt-d, et töödelda sisendandmeid ning klasside maske.

##### Data

Siia tekitab ./Scripts/bands.sh Sentinel-2 piltidest väljundfailid.

##### Label_tifs

Siia tekitab ./Scripts/label_mask.sh klasside rasterfailid.

##### Predictions

Sisaldab ./predict.py poolt tekitatud ennustuse tulemusi ning graafikuid. 

Lisaks sisaldab ennustuste visualiseerimiseks vajalikke käsurea skripte

##### Models 

Sisaldab treenitud mudeleid ning treenimiste metaandmeid.

Lisaks sisaldab mudelite treenimise ajaloo ning metaandmete lugemise skripte 

##### Input_data

Sisaldab ./Scripts/create_input_data.py poolt loodavaid 128x128 sisendtensorite ning tõeväärtuste binaarfaile.

##### Big_tile_data

Sisaldab ./Scripts/create_input_data.py poolt loodavaid 512x512 sisendtensorite ning tõeväärtuste binaarfaile.

Sisaldab skripti kõige rohkemate erinevate klassidega ruudu leidmiseks ning skripti ruudul esinevate klasside loetlemiseks.

##### Scripts

Sisaldab ./Input_data/ ja ./Big_tile_data/ kausta sisendandmete tekitamise skripti ./Scripts/create_input_data.py.

Sisaldab skripte klasside kaalude arvutamiseks: count_class_occurrences.py, calculate_weights.py ning vastavaid

Sisaldab aktivatsioonide visualiseerimise skripti.

## Skriptide jooksutamine käsurealt:

##### ./fit_main.py ning ./generator_main.py:

n: python generator_main.py v51 v50

Loetakse sisse mudel ./Models/v50.h5 ning pärast treenimist salvestatakse mudel failina ./Models.v51.h5

##### ./predict.py:

n: python predict.py v47 150

Loetakse sisse mudel ./Models/v47.h5 ning kasutatakse seda, et teha ennustused 150.ndal 512x512 ruudul. Ennustuste pilt koos tegelike klasside pildiga salvestatakse ./Predictions/fig_v47.npy ning ennustuse statistika salvestatakse ./Predictions/prediction_stats_v47_150.npy

##### ./Predictions/cmd_plot_accuracy_2.py

n: python cmd_plot_accuracy_2.py v47 v48 150

Kuvab mudelite v47 ja v48 ennustuste graafikud 150.ndal 512x512 ruudul

##### ./Predictions/stats_for_2.py

n: python stats_for_2.py v47 v48 150

Kuvab mudelite v47 ja v48 ennustuste statistika 150.ndal 512x512 ruudul


##### ./Models/cmd_plot_hist.py

n: python cmd_plot_hist.py v47

##### ./Models/read_metadata.py

n: python read_metadata.py

##### ./Predictions/plot_accuracy.py

n: python plot_accuracy.py v47

## Vajaliku keskkonna ülesseadmine

##### Rangelt soovitatav on luua omaette virtuaalne pythoni keskkond:

conda create -n environmentName python=3.6.7 anaconda

##### Installida tuleb järgnevad teegid:

conda install ’poppler<0.62’

conda install -c conda-forge ’georaster<1.25’

conda install gdal=2.3.2

conda install -c conda-forge notebook

conda install -c conda-forge keras

conda install -c menpo opencv=3

conda install -c conda-forge matplotlib

conda install pandas

conda install seaborn

conda install scikit-learn

conda install scikit-image

conda install pyshp

conda install shapely

conda install -c conda-forge rasterio

##### Virtuaalse keskkonna kasutamiseks tuleb jooksutada:

source activate environmentName

##### Andmete eeltöötluseks läheb vaja ka töötavat SNAP installatsiooni:

https://step.esa.int/main/snap-6-0-released/
