import numpy as np
import os
import sys
import time

import theano
import theano.tensor as T
import lasagne


data_path = 'spectrograms_reduced.npy'

def load_dataset():
    spectrograms = np.load(data_path)
    spectrograms = spectrograms[:,:10,:] # example, frequency, time
    data = np.empty((spectrograms.shape[0], 1, spectrograms.shape[1], spectrograms.shape[2]))
    for i in range(spectrograms.shape[0]):
            data[i,0,:,:] = spectrograms[i].reshape(1,spectrograms.shape[1],spectrograms.shape[2])
    print(data.shape)
    percent_train_data = 0.9
    n_train_examples = int(percent_train_data*data.shape[0])
    train_data = data[:n_train_examples].astype(np.float32)
    test_data = data[n_train_examples:].astype(np.float32)
    print(train_data.shape)
    print(test_data.shape)
    return train_data, test_data
    
def build_network(input_data=None, input_var=None, n_conv_layers=3, n_filters=1, filter_size=(3,7)):
    
    #####
    ## n_conv_layers=3, filter_size=(3,7)
    
    
    ## convolution dimensions? options:
    ## 1) rectangular filters of size N_FREQUENCY_STEPS x FILTER_SIZE, 1 channel
    ##      a) square filters in the beginning, then a rectangular filter to get a 1D vector representation (do we need it to be 1D? maybe just raveling it works)
    ##          problem: we need to know the output size of the initial squared conv layers in order to set the proper rectangular filter size
    ##      b) rectangular filter in the beginning -> 1D output so then 1D convolutions
    ##          problem: bad representation? too local
    ## 2) n channels = n frequency steps -> 1D convolution from the beginning. probably better, we exploit the full connectivity of channels
    ##
    ##  NOTE: IN CONVNETS CONNECTIVITY IS LOCAL IN SPACE BUT FULL IN DEPTH
    ##          so maybe use time slots as channels? no because it means fixed time... think about solutions for fixed time
    freq_size = input_data.shape[2]
    time_size = input_data.shape[3]
    network = lasagne.layers.InputLayer((None, 1, freq_size, time_size), input_var=input_var)
    ## encoder
    convo_layers = []
    for i in range(n_conv_layers):
        layer = lasagne.layers.Conv2DLayer(
                network, num_filters = n_filters, filter_size=filter_size,
                #~ stride = (2,2),
                nonlinearity=lasagne.nonlinearities.tanh,
                W=lasagne.init.GlorotUniform())
        convo_layers.append(layer)
        network = layer
        print(str(i)+" layer ok")
    
    #~ for j in range(1):
        #~ layer = lasagne.layers.DenseLayer(network, 100, nonlinearity=lasagne.nonlinearities.tanh)
        #~ convo_layers.append(layer)
        #~ network = layer
        #~ print(str(i+j+1)+" layer ok")
    
    bottleneck = network
    
    ## decoder
    for l in reversed(convo_layers):
        #~ network = lasagne.layers.TransposedConv2DLayer(
                #~ network, num_filters = n_filters, filter_size=(filter_size, filter_size), 
                #~ stride = 2,
                #~ W=lasagne.init.GlorotUniform(), 
                #~ nonlinearity=lasagne.nonlinearities.tanh)
        network = lasagne.layers.InverseLayer(network, l)
    return network, bottleneck
    
def iterate_minibatches_autoencoder(inputs, batchsize, shuffle=True):
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt]
        
        
def main(num_epochs, LR, M=0.5, batch_size=256):
    #~ num_epochs = 200
    
    train_data, test_data = load_dataset()
    input_var = T.tensor4('inputs')
    target_var = T.tensor4('targets')
    
    print("Building network...")
    network, bottleneck = build_network(train_data, input_var)

    prediction = lasagne.layers.get_output(network)
    bottleneck_prediction = lasagne.layers.get_output(bottleneck)
    loss = lasagne.objectives.squared_error(prediction, target_var)
    loss = loss.mean()
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.adam(loss, params, learning_rate=LR)
    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    loss_fn = theano.function([input_var, target_var], loss)
    output_fn = theano.function([input_var], prediction)
    bottleneck_activation_fn = theano.function([input_var], bottleneck_prediction)
    
    nparameters = lasagne.layers.count_params(network, trainable=True)
    print("Number of parameters in model: {}".format(nparameters))
    nparams_bn = lasagne.layers.count_params(bottleneck, trainable=True) - lasagne.layers.count_params(lasagne.layers.get_all_layers(bottleneck)[-2], trainable=True)
    print("NUmber of parameters in bottleneck layer: "+str(nparams_bn))
    print("Starting training...")
    
    best_validation_loss = 10  # just a number too large for a loss...
    
    for epoch in range(num_epochs):
        train_loss = 0
        train_batches = 0
        for batch in iterate_minibatches_autoencoder(train_data, batch_size):
            tmp = train_fn(batch, batch)
            #~ print(tmp)
            train_loss += tmp
            train_batches += 1
            #~ print("Batch "+str(train_batches))
        print("Epoch "+str(epoch)+", training loss = "+str(train_loss/train_batches))
        
        test_loss = 0
        test_batches = 0
        for batch in iterate_minibatches_autoencoder(test_data, batch_size):
            tmp = loss_fn(batch, batch)
            #~ print(tmp)
            test_loss += tmp
            test_batches += 1
            #~ print("Batch "+str(train_batches))
        test_loss = test_loss/test_batches
        if best_validation_loss > test_loss:
            best_validation_loss = test_loss
        print("Epoch "+str(epoch)+", test loss = "+str(test_loss/test_batches))
    
    output = output_fn(test_data)
    np.save('out.npy', np.array([output[:,0], test_data[:,0]])) # [0][0]: first index of example, second of (unique) channel
    
    bn_output = bottleneck_activation_fn(test_data)
    print(bn_output.shape)
    # print(bn_output[0])
    
    ## save network:
    np.savez('model.npz', *lasagne.layers.get_all_param_values(network_output))
    
    return best_validation_loss


if __name__ == '__main__':
    if len(sys.argv)>1:
        data_path = sys.argv[1]
    print(data_path)
    main(75, 0.001, batch_size=256)
    
