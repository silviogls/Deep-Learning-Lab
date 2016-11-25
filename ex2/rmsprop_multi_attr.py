
import numpy as np
import os
import time

import theano
import theano.tensor as T
import lasagne


'''
Image arrays have the shape (N, 3, 32, 32), where N is the size of the
corresponding set. This is the format used by Lasagne/Theano. To visualize the
images, you need to change the axis order, which can be done by calling
np.rollaxis(image_array[n, :, :, :], 0, start=3).

Each image has an associated 40-dimensional attribute vector. The names of the
attributes are stored in self.attr_names.
'''

data_path = "/home/lmb/Celeb_data"


class Network:
    
    def __init__(self):
        self.network = None
        self.train_images = None
        self.batch_size = None

    def load_data(self):
        self.train_images = np.float32(np.load(os.path.join(
                data_path, "train_images_32.npy"))) / 255.0
        self.train_labels = np.uint8(np.load(os.path.join(
                data_path, "train_labels_32.npy")))
        self.val_images = np.float32(np.load(os.path.join(
                data_path, "val_images_32.npy"))) / 255.0
        self.val_labels = np.uint8(np.load(os.path.join(
                data_path, "val_labels_32.npy")))
        self.test_images = np.float32(np.load(os.path.join(
                data_path, "test_images_32.npy"))) / 255.0
        self.test_labels = np.uint8(np.load(os.path.join(
                data_path, "test_labels_32.npy")))
        
        with open(os.path.join(data_path, "attr_names.txt")) as f:
            self.attr_names = f.readlines()[0].split()
        
       
    
    # TODO: set strides and padding
    def build_network(self, input_data):
        N_FILTERS = 16
        FILTER_SIZE = 7
        DENSE_UNITS_1 = 10
        DENSE_UNITS_2 = 10
        
        # input layer
        network = lasagne.layers.InputLayer((None, 3, 32, 32), input_var=input_data)
        
        # convolutions + pooling
        conv_layer_1 = lasagne.layers.Conv2DLayer(network, num_filters=N_FILTERS, pad = 'same', filter_size=(FILTER_SIZE, FILTER_SIZE), nonlinearity=lasagne.nonlinearities.tanh, W=lasagne.init.GlorotUniform())
        network = lasagne.layers.MaxPool2DLayer(conv_layer_1, pool_size=(2, 2))
        conv_layer_2 = lasagne.layers.Conv2DLayer(network, num_filters=N_FILTERS, pad = 'same', filter_size=(FILTER_SIZE, FILTER_SIZE), nonlinearity=lasagne.nonlinearities.tanh, W=lasagne.init.GlorotUniform())
        network = lasagne.layers.MaxPool2DLayer(conv_layer_2, pool_size=(3, 3))
        conv_layer_3 = lasagne.layers.Conv2DLayer(network, num_filters=N_FILTERS, pad = 'same', filter_size=(FILTER_SIZE, FILTER_SIZE), nonlinearity=lasagne.nonlinearities.tanh, W=lasagne.init.GlorotUniform())
        network = lasagne.layers.MaxPool2DLayer(conv_layer_3, pool_size=(3, 3))
        
        # output layers
        output_layers = [None]*40
        for i in range(40):            
            # fully connected layers
            output_layers[i] = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, p=.5), num_units=DENSE_UNITS_1, nonlinearity=lasagne.nonlinearities.tanh)
            output_layers[i] = lasagne.layers.DenseLayer(lasagne.layers.dropout(output_layers[i], p=.5), num_units=DENSE_UNITS_2, nonlinearity=lasagne.nonlinearities.tanh)
            output_layers[i] = lasagne.layers.DenseLayer(output_layers[i], num_units=1, nonlinearity=lasagne.nonlinearities.sigmoid)
        
        #
        self.conv_layers = [conv_layer_1, conv_layer_2]
        self.network = output_layers
        
        print("n of filters: "+str(N_FILTERS))
        print("filters size: "+str(FILTER_SIZE))
        print("n of units in fully connected layers: "+str(DENSE_UNITS_1))

        return network

    
    def batches(self, X, Y, batch_size):
        N_OF_ATTRIBUTES = 40
        train_idxs = np.random.permutation(X.shape[0])
        for i in range(0, X.shape[0], batch_size):
            X_batch = X[train_idxs[i:i+batch_size]]
            Y_batch = Y[train_idxs[i:i+batch_size]].reshape(-1, N_OF_ATTRIBUTES)
            yield X_batch, Y_batch


    # saves the filters of the first convolutional layer as png images
    def get_conv_filters(self, id):
        import math
        from PIL import Image

        params = self.conv_layers[0].get_params()[0].get_value()  #I guess get_params()[1] is the bias, TODO: sum the bias...
        print(params.shape)
        
        side = int(math.sqrt(params.shape[0]))
        filter_size = params.shape[2]
        filter_matrix = Image.new('RGB', (side*(filter_size+1), side*(filter_size+1)))
        for c in range(side):
            for r in range(side):
                
                filter_matrix.paste( 
                    Image.fromarray( np.moveaxis((params[r*side+c, :, :, :]*255).astype('uint8'), 0, 2), 'RGB' ), 
                    (r*(filter_size+1), c*(filter_size+1)) )
                    
        filter_matrix.save("filters/filter_"+str(id)+".png")
            
    
    # TODO: optimization scheme choice with parameter?
    def train(self, max_epochs, batch_size):
        
        if self.train_images is None:
            print("Need to load the data first")
            return
        
        input_data = T.tensor4('inputs')
        labels = T.matrix('labels')
        
        self.build_network(input_data)
        
        # predictions
        predictions = lasagne.layers.get_output(self.network)
        p_fn = theano.function([input_data], predictions)
        predictions_test = lasagne.layers.get_output(self.network, deterministic=True)
        
        # performance metrics:
        #losses = theano.shared(np.zeros(40))
        #losses_test = theano.shared(np.zeros(40))
        losses = [0]*40
        losses_test = [0]*40
        loss = 0
        loss_test = 0
        test_accuracies = [0]*40
    
        
        
        for i in range(40):
            losses[i] = lasagne.objectives.binary_crossentropy(predictions[i].T, labels[:,i]).mean()
            losses_test[i] = lasagne.objectives.binary_crossentropy(predictions_test[i].T, labels[:,i]).mean()
            test_accuracies[i] = (T.eq(T.round(predictions_test[i].T), labels[:,i])).mean()
            loss += losses[i]

            
        # accuracy  
        
        
        params = lasagne.layers.get_all_params(self.network, trainable=True)
        train_function = theano.function([input_data, labels], losses, updates=lasagne.updates.rmsprop(loss, params, learning_rate=0.0005))
        validation_function = theano.function( [input_data, labels], losses_test ) 
        validation_accuracy = theano.function( [input_data, labels], test_accuracies )
        
        print("Traning progress:")
        print("(epoch, time, training error, validation error, validation accuracy)")
        TRAINING_SET_SIZE =   100000
        VALIDATION_SET_SIZE = 50000
        for epoch in range(max_epochs):
            tr_losses = [0]*40
            training_loss = 0
            count = 0
            num_of_training_batches = 0
            start_time = time.time()
            for input_batch, labels_batch in self.batches(self.train_images[:TRAINING_SET_SIZE,:,:,:], self.train_labels[:TRAINING_SET_SIZE,:], batch_size):
                start_time_batch = time.time()
                
                #pr = p_fn(input_batch)
                #for p in pr:
                    #if np.any(p<=0): 
                        #print("NEGATIVE PREDICTION")
                        #return
                    #if np.any(np.isnan(p)): 
                        #print("NaN PREDICTION")
                        #return
                
                
                tmp_losses = train_function(input_batch, labels_batch)
                tr_losses = [sum(x) for x in zip(tr_losses, tmp_losses)]
               
                
                #pm = lasagne.layers.Layer.get_params(self.conv_layers[-1], trainable=True)[0].get_value()
                #if np.any(np.isnan(pm)): 
                    #print("NaN PARAMETERS")
                    #return
                
                    
                #training_loss += new_loss
                count += 1
                #print(int(time.time()-start_time_batch))
            training_errors = np.array(tr_losses)/count
            
            
            count = 0
            val_losses = [0]*40
            val_accuracies = [0]*40
            for input_batch, labels_batch in self.batches(self.val_images[:VALIDATION_SET_SIZE,:,:,:], self.val_labels[:VALIDATION_SET_SIZE,:], batch_size):
                tmp_losses = validation_function(input_batch, labels_batch)
                tmp_accuracies = validation_accuracy(input_batch, labels_batch)
                val_losses = [sum(x) for x in zip(val_losses, tmp_losses)]
                val_accuracies = [sum(x) for x in zip(val_accuracies, tmp_accuracies)]
                count += 1
            validation_errors = np.array(val_losses)/count
            validation_accuracies = np.array(val_accuracies)/count*100
            
            if epoch%4 == 1: self.get_conv_filters(epoch)
            #print("{} of {}, {:d} , {:.6f}, {:.6f} , {:.6f}".format(epoch+1, max_epochs, int(time.time()-start_time), training_errors, validation_errors, validation_accuracy/count*100))
            print('epoch  '+str(epoch))
            print(training_errors, sep=' ')
            print(validation_errors, sep=' ')
            print(validation_accuracies, sep=' ')
            print('GENDER attribute validation accuracy: '+str(validation_accuracies[20]))
        self.get_conv_filters("final")
        #~ print("Test error: {:.6f}".format(validation_function(input_batch, labels_batch)))
                
net = Network()
net.load_data()
net.train(60, 256)
