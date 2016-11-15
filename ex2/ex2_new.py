

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
        network = lasagne.layers.InputLayer((None, 3, 32, 32), input_var=input_data)
        conv_layer_1 = lasagne.layers.Conv2DLayer(network, num_filters=16, pad = 'same', filter_size=(5, 5), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
        network = lasagne.layers.MaxPool2DLayer(conv_layer_1, pool_size=(2, 2))
        conv_layer_2 = lasagne.layers.Conv2DLayer(network, num_filters=16, pad = 'same', filter_size=(5, 5), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
        network = lasagne.layers.MaxPool2DLayer(conv_layer_2, pool_size=(2, 2))
        #network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, p=.5), num_units=50, nonlinearity=lasagne.nonlinearities.rectify)
        network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, p=.5), num_units=100, nonlinearity=lasagne.nonlinearities.rectify)
        network = lasagne.layers.DenseLayer(network, num_units=20, nonlinearity=lasagne.nonlinearities.softmax)
        self.network = network
        self.conv_layers = [conv_layer_1, conv_layer_2]
        
        return network
        
    def predict(self, deterministic = None):
        if deterministic is not None:
            return lasagne.layers.get_output(self.network, deterministic=deterministic)
        else:
            return lasagne.layers.get_output(self.network)
        
    def loss(self, labels):
        return lasagne.objectives.categorical_crossentropy(self.predict(), labels).mean()
    
    # loss for test error computation (difference is deterministic flag)
    # could merge this with loss()
    def loss_test(self, labels):
        return lasagne.objectives.categorical_crossentropy(self.predict(deterministic=True), labels).mean()
    
    def test_accuracy(self, labels):
        return T.mean(T.eq(T.argmax(self.predict(deterministic=True), axis=1), labels), dtype=theano.config.floatX)
    
    # TODO: add more optimization schemes and handle exceptions
    def updates(self, optimization_scheme, loss, learning_rate):
        params = lasagne.layers.get_all_params(self.network, trainable=True)
        if optimization_scheme == 'sgd':
            return lasagne.updates.sgd(loss, params, learning_rate=learning_rate)
        if optimization_scheme == 'momentum':
            return lasagne.updates.momentum(loss, params, learning_rate=learning_rate)
        if optimization_scheme == 'rmsprop':
            return lasagne.updates.sgd(loss, params, learning_rate=learning_rate)
        else:
            pass
    
    def batches(self, X, Y, batch_size):
        train_idxs = np.random.permutation(X.shape[0])
        for i in range(0, X.shape[0], batch_size):
            X_batch = X[train_idxs[i:i+batch_size]]
            Y_batch = Y[train_idxs[i:i+batch_size]]
            yield X_batch, Y_batch


    # saves the filters of the convolutional layers as png images
    def get_conv_filters(self):
        import math
        from PIL import Image
        for (i, cl) in enumerate(self.conv_layers):
            params = cl.get_params()[0].get_value()  #I guess get_params()[1] is the bias, TODO: sum the bias...
            print(params.shape)
            for channel in range(params.shape[1]):
                #filter = (np.moveaxis(params[f,:,:,:], 0, 2)*255).astype('uint8')
                side = int(math.sqrt(params.shape[0]))
                filter_rows  = []
                for c in range(side):
                    filters = []
                    for r in range(side):
                        filters.append((params[r*side+c, channel, :, :]*255).astype('uint8'))
                    filter_rows.append(np.concatenate(np.array(filters), axis=0))
                filter_matrix = np.concatenate(np.array(filter_rows), axis=1)
                Image.fromarray(filter_matrix).save("filters/filter_"+str(channel)+".png")
            
    
    # TODO: optimization scheme choice with parameter?
    def train(self, max_epochs, batch_size):
        
        if self.train_images is None:
            print("Need to load the data first")
            return
        
        input_data = T.tensor4('inputs')
        labels = T.ivector('labels')
        self.build_network(input_data)
        loss = lasagne.objectives.categorical_crossentropy(lasagne.layers.get_output(self.network), labels).mean()
        loss_test = lasagne.objectives.categorical_crossentropy(lasagne.layers.get_output(self.network, deterministic=True), labels).mean()
        test_accuracy = T.mean(T.eq(T.argmax(lasagne.layers.get_output(self.network, deterministic=True), axis=1), labels), dtype=theano.config.floatX)
        
        params = lasagne.layers.get_all_params(self.network, trainable=True)
        #train_function = theano.function([input_data, labels], loss, updates=self.updates('rmsprop', loss, 0.3))
        train_function = theano.function([input_data, labels], loss, updates=lasagne.updates.sgd(loss, params, learning_rate=0.03))
        validation_function = theano.function([input_data, labels], [loss_test, test_accuracy] )   # good?
        
        print("Traning progress:")
        print("(epoch, time, training error, validation error, validation accuracy)")
        TRAINING_SET_SIZE = 200000
        VALIDATION_SET_SIZE = 50000
        for epoch in range(max_epochs):
            training_loss = 0
            count = 0
            num_of_training_batches = 0
            start_time = time.time()
            for input_batch, labels_batch in self.batches(self.train_images[:TRAINING_SET_SIZE,:,:,:], self.train_labels[:TRAINING_SET_SIZE,20], batch_size):
                start_time_batch = time.time()
                training_loss += train_function(input_batch, labels_batch)
                count += 1
                #print(int(time.time()-start_time_batch))
            training_error = training_loss/count
            
            count = 0
            validation_loss = 0
            validation_accuracy = 0
            for input_batch, labels_batch in self.batches(self.val_images[:VALIDATION_SET_SIZE,:,:,:], self.val_labels[:VALIDATION_SET_SIZE,20], batch_size):
                val_err, val_acc = validation_function(input_batch, labels_batch)
                validation_accuracy += val_acc
                validation_loss += val_err
                count += 1
            validation_error = validation_loss/count
            
            #if epoch%5 == 0: self.get_conv_filters()
            print("{} of {}, {:d} , {:.6f}, {:.6f} , {:.6f}".format(epoch+1, max_epochs, int(time.time()-start_time), training_error, validation_error, validation_accuracy/count*100))
        
        #self.get_conv_filters()
        #~ print("Test error: {:.6f}".format(validation_function(input_batch, labels_batch)))
                
net = Network()
net.load_data()
net.train(30, 100)
