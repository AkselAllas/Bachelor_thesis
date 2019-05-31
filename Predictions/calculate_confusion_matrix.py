#!/usr/bin/env python
# coding: utf-8

import numpy as numpy
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

from get_fig_data import *

def plot_confusion_matrix(class_names, version, y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    #if not title:
    #    if normalize:
    #        title = 'Normeeritud confusion matrix '+version
    #    else:
    #        title = 'Confusion matrix, without normalization '+version

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    classes = classes[unique_labels(y_pred)]
    class_list = []
    for element in classes:
        class_list.append(str(int(element))+"-"+class_names[element])
    
    print(class_list)
    input("Press Enter to continue...")

    #print(cm.shape)
    #print(cm[0])
    #input("Press Enter to continue...")

    true_labels = unique_labels(y_true)    
    classes_cm_index = np.zeros((len(classes))).astype(int)
    for i in np.arange(len(classes)):
        classes_cm_index[i] = np.where(true_labels==classes[i])[0][0] 

    new_cm = np.zeros((len(classes),len(classes))).astype(int)
    for i in range(len(classes)):
        for j in range(len(classes)):
            new_cm[i][j] = cm[classes_cm_index[i]][classes_cm_index[j]]

    if normalize:
        new_cm = new_cm.astype('float') / new_cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')


    fig, ax = plt.subplots()
    im = ax.imshow(new_cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...

    #ax.set(xticks=np.arange(cm.shape[1]),
    #       yticks=np.arange(cm.shape[0]),


    ax.set(xticks=np.arange(len(classes)),
           yticks=np.arange(len(classes)),
           # ... and label them with the respective list entries
           xticklabels=class_list, yticklabels=class_list,
           title=title,
           ylabel='Tegelik klass',
           xlabel='Ennustatud klass')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = new_cm.max() / 2.
    k=0
    for i in range(len(classes_cm_index)):
        l=0
        for j in range(len(classes_cm_index)):
            ax.text(j, i, format(new_cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if new_cm[i, j] > thresh else "black")
            l+=1
        k+=1
    fig.tight_layout()
    return ax

class_names = np.array([
"Mitte-põllukultuur"
,"Aedmaasikas"
,"Astelpaju"
,"Heintaimed, kõrrelised"
,"Heintaimed, liblikõielised"
,"Kaer"
,"Kanep"
,"Karjatamine"
,"Kartul"
,"Mais"
,"Mustkesa"
,"Muu"
,"Põldhernes"
,"Põlduba"
,"Peakapsas"
,"Porgand"
,"Punapeet"
,"Rukis"
,"Sööti jäetud maa"
,"Suvinisu ja speltanisu"
,"Suvioder"
,"Suviraps ja -rüps"
,"Suvitritikale"
,"Talinisu"
,"Talioder"
,"Taliraps ja -rüps"
,"Talitritikale"
,"Tatar"
])

classes=np.array(range(0,28))
#print(type(classes[0]))

version = str(sys.argv[1])
tile_list = ['0', '1','120', '2', '10', '20', '80', '100',  '160', '170']
Y_true = np.array(np.zeros((0))).astype(int)
Y_pred = np.array(np.zeros((0))).astype(int)

for tile in tile_list:
    Y_ground_truth, Y_predictions = get_fig_data(version,tile)

    Y_true = np.concatenate((Y_true, (np.reshape(Y_ground_truth,(512*512)).astype(int))))
    Y_pred = np.concatenate((Y_pred, (np.reshape(Y_predictions, (512*512)).astype(int))))

#print(Y_true.shape)
#input("Press Enter to continue...")

#print(type(Y_true[0]))
#print(Y_pred[0])

plt.show(plot_confusion_matrix(class_names, version, Y_true,Y_pred,classes,False))
