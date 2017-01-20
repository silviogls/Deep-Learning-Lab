import numpy as np
import os
import sys
import time

import theano
import theano.tensor as T
import lasagne
import utils

data_path = 'spectrograms_reduced.npy'

        
def main_fully_connected(num_epochs, LR, M=0.9, batch_size=256):
    #~ num_epochs = 200
    
    train_data, test_data = utils.load_dataset_zipped(data_path, network_type = 'fully connected')
    test_data = np.concatenate(test_data)
    input_var = T.matrix('inputs')
    target_var = T.matrix('targets')
    
    print("Building network...")
    network, bottleneck = utils.build_network(train_data, input_var)

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
        for batch in utils.iterate_minibatches_autoencoder(train_data, batch_size):
            tmp = train_fn(batch, batch)
            #~ print(tmp)
            train_loss += tmp
            train_batches += 1
            #~ print("Batch "+str(train_batches))
        #print("Epoch "+str(epoch)+", training loss = "+str(train_loss/train_batches))
        
        test_loss = 0
        test_batches = 0
        for batch in utils.iterate_minibatches_autoencoder(test_data, batch_size):
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
    np.save('out.npy', np.array([output, test_data])) # [0][0]: first index of example, second of (unique) channel
    
    bn_output = bottleneck_activation_fn(test_data)
    print("bottleneck size: "+str(bn_output.shape))
    # print(bn_output[0])
    
    ## save network:
    np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
    
    return best_validation_loss

def main_convolutional(num_epochs, LR, M=0.9, batch_size=256):
    #~ num_epochs = 200
    
    train_data, test_data = utils.load_dataset_zipped(data_path, train_size = .75, network_type = 'convolutional')
    test_data = np.concatenate(test_data)
    input_var = T.tensor4('inputs')
    target_var = T.tensor4('targets')
    
    print("Building network...")
    network, bottleneck = utils.build_network_convolutional(train_data, input_var)

    # l2 regularizer
    l2_penalty = lasagne.regularization.regularize_layer_params(network, lasagne.regularization.l2)*1e-1

    prediction = lasagne.layers.get_output(network)
    bottleneck_prediction = lasagne.layers.get_output(bottleneck)
    #loss = lasagne.objectives.squared_error(prediction, target_var) + l2_penalty
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
    
    # debug output shape
    #out = output_fn(train_data[:20])
    #print(out.shape)
    
    for epoch in range(num_epochs):
        train_loss = 0
        train_batches = 0
        for batch in utils.iterate_minibatches_autoencoder(train_data, batch_size):
            tmp = train_fn(batch, batch)
            #~ print(tmp)
            train_loss += tmp
            train_batches += 1
            #~ print("Batch "+str(train_batches))
        #print("Epoch "+str(epoch)+", training loss = "+str(train_loss/train_batches))
        
        test_loss = 0
        test_batches = 0
        for batch in utils.iterate_minibatches_autoencoder(test_data, batch_size):
            tmp = loss_fn(batch, batch)
            #~ print(tmp)
            test_loss += tmp
            test_batches += 1
            #~ print("Batch "+str(train_batches))
        test_loss = test_loss/test_batches
        if best_validation_loss > test_loss:
            best_validation_loss = test_loss
        print("Epoch "+str(epoch)+", test loss = "+str(test_loss/test_batches))
        results = open("results.csv", 'a')
        results.write(str(epoch)+", "+str(test_loss/test_batches)+"\n")
        
    ## save encoding network:
    print("saving encoding network...")
    np.savez('encoder.npz', *lasagne.layers.get_all_param_values(bottleneck))
    
    print("saving reconstruction examples of test set...")
    output = output_fn(test_data[:2])
    np.save('out.npy', np.array([output[:,0], test_data[:,0]])) # [0][0]: first index of example, second of (unique) channel
    
    bn_output = bottleneck_activation_fn(test_data)
    print("bottleneck activation size: "+str(bn_output.shape))
    # print(bn_output[0])
    
    
    return best_validation_loss


if __name__ == '__main__':
    if len(sys.argv)>1:
        data_path = sys.argv[1]
    print(data_path)
    main_convolutional(200, 0.00005, batch_size=256)
    
