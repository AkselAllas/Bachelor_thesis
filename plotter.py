import matplotlib
import matplotlib.pyplot as plt
import numpy as np

class plotter:

    def __init__(self, s2_preprocessor=None, s2_model=None):
        self.s2_preprocessor = s2_preprocessor
        self.s2_model = s2_model

    def plot_labels(self, labels):
        f, arr = plt.subplots()
        img = arr.imshow(labels,vmin=0,vmax=self.s2_preprocessor.nb_classes)
        cig = f.colorbar(img)
        plt.show()
   
    def plot_tile(self, label_map_tiles, tile_location): 
        plt.imshow(label_map_tiles[tile_location[0]][tile_location[1]])
        plt.show()

    def plot_model_result(self, hist):
        # Plot the results
        train_loss=hist.history['loss']
        val_loss=hist.history['val_loss']
        train_acc=hist.history['acc']
        val_acc=hist.history['val_acc']
        mse=hist.history['mean_squared_error']
        xc=range(s2_model.nb_epochs)

        plt.figure(1,figsize=(7,5))
        plt.plot(xc,train_loss)
        plt.plot(xc,val_loss)
        plt.xlabel('num of Epochs')
        plt.ylabel('loss')
        plt.title('train_loss vs val_loss')
        plt.grid(True)
        plt.legend(['train','val'])
        #print plt.style.available # use bmh, classic,ggplot for big pictures
        plt.style.use(['classic'])

        plt.figure(2,figsize=(7,5))
        plt.plot(xc,train_acc)
        plt.plot(xc,val_acc)
        plt.xlabel('num of Epochs')
        plt.ylabel('accuracy')
        plt.title('train_acc vs val_acc')
        plt.grid(True)
        plt.legend(['train','val'],loc=4)
        #print plt.style.available # use bmh, classic,ggplot for big pictures
        plt.style.use(['classic'])
        plt.show()

    def plot_model_prediction(self, y_predictions, tile_location):
        prediction_map = np.zeros((s2_preprocessor.tile_dimension,s2_preprocessor.tile_dimension))
        count=0
        for i in range(s2_preprocessor.tile_dimension):
            for j in range(s2_preprocessor.tile_dimension):
                    prediction_map[i][j] = y_predictions[s2_preprocessor.tile_dimension*i+j]
                    if prediction_map[i][j]==label_tulemus[tile_location[0]][tile_location[1]][i][j]:
                        count+=1
        print("Accuracy is:")
        print(count/(s2_preprocessor.tile_dimension*s2_preprocessor.tile_dimension))
        plt.imshow(prediction_map)
        plt.show()
