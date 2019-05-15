import matplotlib
import matplotlib.pyplot as plt
import numpy as np

class plotter:

    def __init__(self, s2_preprocessor, cmap='viridis'):
        self.s2_preprocessor = s2_preprocessor
        self.cmap = cmap

    def plot_label_map(self, label_map):
        f, arr = plt.subplots()
        img = arr.imshow(label_map,vmin=0,vmax=self.s2_preprocessor.nb_classes, cmap=self.cmap)
        cig = f.colorbar(img)
        plt.show()
   
    def plot_tile(self, label_map_tiles, tile_location): 
        f, arr = plt.subplots()
        img = arr.imshow(label_map_tiles[tile_location[0]][tile_location[1]],vmin=0,vmax=self.s2_preprocessor.nb_classes,cmap=self.cmap)
        cig = f.colorbar(img)
        plt.show()


    def plot_2D_tile(self, input_2D_tile): 
        f, arr = plt.subplots()
        img = arr.imshow(input_2D_tile, cmap=self.cmap)
        cig = f.colorbar(img)
        plt.show()

    def plot_model_result(self, hist):
        # Plot the results
        train_loss=hist.history['loss']
        val_loss=hist.history['val_loss']
        train_acc=hist.history['acc']
        val_acc=hist.history['val_acc']
        mse=hist.history['mean_squared_error']
        xc=range(len(train_loss))

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

    def plot_model_prediction(self, y_predictions, tile_location, label_map_tiles):
        prediction_map = np.zeros((self.s2_preprocessor.tile_dimension,self.s2_preprocessor.tile_dimension))
        f, axarr = plt.subplots(2,2)
        count=0
        for i in range(self.s2_preprocessor.tile_dimension):
            for j in range(self.s2_preprocessor.tile_dimension):
                    prediction_map[i][j] = y_predictions[self.s2_preprocessor.tile_dimension*i+j]
                    if(prediction_map[i][j]==label_map_tiles[tile_location[0]][tile_location[1]][i][j]):
                        count+=1
        print("Accuracy is:")
        print(count/(self.s2_preprocessor.tile_dimension*self.s2_preprocessor.tile_dimension))
        axarr[0][0].imshow(prediction_map)
        axarr[0][1].imshow(label_map_tiles[tile_location[0]][tile_location[1]])
        plt.show()

    def plot_model_prediction_v2(self, y_predictions, true_values, name_arg, predictions_path):
        prediction_map = np.zeros((self.s2_preprocessor.tile_dimension,self.s2_preprocessor.tile_dimension))
        f, axarr = plt.subplots(2,2)
        count=0
        labels_map = np.zeros((self.s2_preprocessor.tile_dimension,self.s2_preprocessor.tile_dimension))
        for i in range(int(self.s2_preprocessor.tile_dimension)):
            for j in range(self.s2_preprocessor.tile_dimension):
                    labels_map[i][j] = np.argmax(true_values[self.s2_preprocessor.tile_dimension*i+j])
                    if(y_predictions[self.s2_preprocessor.tile_dimension*i+j]==np.argmax(true_values[self.s2_preprocessor.tile_dimension*i+j])):
                        count+=1

        Y_predictions = np.zeros((self.s2_preprocessor.tile_dimension,self.s2_preprocessor.tile_dimension))
        for i in range(int(self.s2_preprocessor.tile_dimension)):
            for j in range(self.s2_preprocessor.tile_dimension):
                    Y_predictions[i][j] = y_predictions[self.s2_preprocessor.tile_dimension*i+j]
        true_val_non_hot = [np.where(r==1)[0][0] for r in true_values]
        axarr[0][0].imshow(labels_map)
        axarr[0][1].imshow(Y_predictions)
        np.save(predictions_path+'fig'+name_arg+'.npy', f)
        #pickle.dump(fig, open('FigureObject.fig.pickle', 'wb'))

    def plot_input_vs_labels(self, label_map_tiles, tile_location, input_data):
        f, axarr = plt.subplots(2,2)


        input_data_map = np.zeros((self.s2_preprocessor.tile_dimension,self.s2_preprocessor.tile_dimension))
        for i in range(self.s2_preprocessor.tile_dimension):
            for j in range(self.s2_preprocessor.tile_dimension):
                    input_data_map[i][j] = input_data[self.s2_preprocessor.tile_dimension*i+j][5][2][2][0]

        axarr[0][0].imshow(input_data_map)
        axarr[0][1].imshow(label_map_tiles[tile_location[0]][tile_location[1]])
        plt.show()


    def plot_input_vs_labels_v2(self, one_hot_labels, input_data):
        f, axarr = plt.subplots(2,2)

        input_map  = np.zeros((self.s2_preprocessor.tile_dimension,self.s2_preprocessor.tile_dimension))
        for i in range(int(self.s2_preprocessor.tile_dimension/2)):
            for j in range(self.s2_preprocessor.tile_dimension):
                    input_map[i][j] = input_data[self.s2_preprocessor.tile_dimension*i+j][0][0][0][0]

        labels_map  = np.zeros((self.s2_preprocessor.tile_dimension,self.s2_preprocessor.tile_dimension))
        for i in range(int(self.s2_preprocessor.tile_dimension/2)):
            for j in range(self.s2_preprocessor.tile_dimension):
                    labels_map[i][j] = np.argmax(one_hot_labels[self.s2_preprocessor.tile_dimension*i+j])

        axarr[0][0].imshow(input_map)
        axarr[0][1].imshow(labels_map)
        plt.show()

    def plot_input_vs_labels_v3(self, input_map, labels_map):
        f, axarr = plt.subplots(2,2)

        print(input_map)
        print(labels_map)

        axarr[0][0].imshow(input_map)
        axarr[0][1].imshow(labels_map)
        plt.show()

    def plot_labels(self, labels):
        f, arr = plt.subplots()

        labels_map  = np.zeros((self.s2_preprocessor.tile_dimension,self.s2_preprocessor.tile_dimension))
        for i in range(self.s2_preprocessor.tile_dimension):
            for j in range(self.s2_preprocessor.tile_dimension):
                    labels_map[i][j] = labels[self.s2_preprocessor.tile_dimension*i+j]

        img = arr.imshow(labels_map,vmin=0,vmax=self.s2_preprocessor.nb_classes,cmap=self.cmap)
        cig = f.colorbar(img)
        plt.show()

    
    def plot_one_hot_labels(self, one_hot_labels):
        f, arr = plt.subplots()

        labels_map  = np.zeros((self.s2_preprocessor.tile_dimension,self.s2_preprocessor.tile_dimension))
        for i in range(self.s2_preprocessor.tile_dimension):
            for j in range(self.s2_preprocessor.tile_dimension):
                    labels_map[i][j] = np.argmax(one_hot_labels[self.s2_preprocessor.tile_dimension*i+j])

        img = arr.imshow(labels_map,vmin=0,vmax=self.s2_preprocessor.nb_classes,cmap=self.cmap)
        cig = f.colorbar(img)
        plt.show()
