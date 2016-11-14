

import numpy as np
import os

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
        
        self.batch_size = 200
    
    def build_network(self, input_data):
        network = lasagne.layers.InputLayer((None, 3, 32, 32), input_var=input_data)
        conv_layer_1 = lasagne.layers.Conv2DLayer(network, num_filters=3, filter_size=(10, 10), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
        network = lasagne.layers.MaxPool2DLayer(conv_layer_1, pool_size=(2, 2))
        conv_layer_2 = lasagne.layers.Conv2DLayer(network, num_filters=3, filter_size=(10, 10), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
        network = lasagne.layers.MaxPool2DLayer(conv_layer_2, pool_size=(2, 2))
        network = lasagne.layers.DenseLayer(network, num_units=50, nonlinearity=lasagne.nonlinearities.softmax)
        network = lasagne.layers.DenseLayer(network, num_units=25, nonlinearity=lasagne.nonlinearities.softmax)
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
    
    
    def updates(self, optimization_scheme, loss, learning_rate):
        params = lasagne.layers.get_all_params(self.network, trainable=True)
        if optimization_scheme == 'sgd':
            return lasagne.updates.sgd(loss, params, learning_rate=learning_rate)
        else:
            #handle exception... TODO
            pass
    
    def batches(self, X, Y, batch_size):
        train_idxs = np.random.permutation(X.shape[0])
        for i in range(0, X.shape[0], batch_size):
            X_batch = X[train_idxs[i:i+batch_size]]
            Y_batch = Y[train_idxs[i:i+batch_size]]
            yield X_batch, Y_batch

    def get_conv_filters(self):
        from PIL import Image
        for (i, cl) in enumerate(self.conv_layers):
            params = cl.get_params()[0].get_value()  #I guess get_params()[1] is the bias, TODO: sum the bias...
            print(params.shape)
            for f in range(params.shape[0]):
                filter = (np.moveaxis(params[f,:,:,:], 0, 2)*255).astype('uint8')
                #print(filter)
                #print(filter.shape)
                Image.fromarray(filter).save("filter_"+str(i)+"_"+str(f)+".png")
            
            
    
    # TODO: optimization scheme choice with parameter?
    def train(self, max_epochs):
        
        if self.train_images is None:
            print("Need to load the data first")
            return
        
        input_data = T.tensor4('inputs')
        labels = T.ivector('labels')
        self.build_network(input_data)
        loss = self.loss(labels)
        loss_test = self.loss_test(labels)
        train_function = theano.function([input_data, labels], loss, updates=self.updates('sgd', loss, 0.3))
        validation_function = theano.function([input_data, labels], loss_test)   # good?
        
        print("Traning progress:")
        print("(epoch, training error, validation error)")
        for epoch in range(max_epochs):
            training_loss = 0
            count = 0
            num_of_training_batches = 0
            for input_batch, labels_batch in self.batches(self.train_images[:10000,:,:,:], self.train_labels[:,20], self.batch_size):
                training_loss += train_function(input_batch, labels_batch)
                count += 1
            training_error = training_loss/count
            
            count = 0
            validation_loss = 0
            for input_batch, labels_batch in self.batches(self.val_images[:5000,:,:,:], self.val_labels[:,20], self.batch_size):
                validation_loss += validation_function(input_batch, labels_batch)
                count += 1
            validation_error = validation_loss/count
            #self.get_conv_filters()
            print("{} of {}, {:.6f}, {:.6f}".format(epoch+1, max_epochs, training_error, validation_error))
        
        self.get_conv_filters()
        #~ print("Test error: {:.6f}".format(validation_function(input_batch, labels_batch)))
                
net = Network()
net.load_data()
net.train(30)

