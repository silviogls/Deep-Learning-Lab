import numpy as np
import os
import sys
import time

import theano
import theano.tensor as T
import lasagne
import utils
import matplotlib.pyplot as plt

data_path = 'spectrograms_reduced.npy'


def train(num_epochs, LR, M=0.9, batch_size=256):    
    train_data, _, test_data, _, _, _ = utils.load_dataset(data_path)
    #~ test_data = np.concatenate(test_data)
    input_var = T.tensor4('inputs')
    target_var = T.tensor4('targets')
    
    print("Building network...")
    network, bottleneck = utils.build_autoencoder(train_data, input_var)

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
    
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.squared_error(test_prediction, target_var)
    test_loss = test_loss.mean()
    val_fn = theano.function([input_var, target_var], test_loss)
    
    nparameters = lasagne.layers.count_params(network, trainable=True)
    print("Number of parameters in model: {}".format(nparameters))
    nparams_bn = lasagne.layers.count_params(bottleneck, trainable=True) - lasagne.layers.count_params(lasagne.layers.get_all_layers(bottleneck)[-2], trainable=True)
    print("NUmber of parameters in bottleneck layer: "+str(nparams_bn))
    print("Starting training...")
    
    best_validation_loss = 10  # just a number too large for a loss...
    r = open("results.csv", "w") # just open and close the file to wipe it clean
    r.close()
    
    for epoch in range(num_epochs):
        train_loss = 0
        train_batches = 0
        for batch in utils.iterate_minibatches_autoencoder(train_data, batch_size):
            tmp = train_fn(batch, batch)
            train_loss += tmp
            train_batches += 1
        
        test_loss = 0
        test_batches = 0
        for batch in utils.iterate_minibatches_autoencoder(test_data, batch_size):
            tmp = val_fn(batch, batch)
            #~ print(tmp)
            test_loss += tmp
            test_batches += 1

        if best_validation_loss > test_loss:
            best_validation_loss = test_loss
        print("Epoch "+str(epoch)+", test loss = "+str(test_loss/test_batches))
        results = open("results.csv", 'a')
        results.write(str(epoch)+", "+str(test_loss/test_batches)+"\n")
        results.close()
        
        save_tmp = True
        if save_tmp and epoch%5==4:
            ## save encoding network:
            print("backing up encoding network...")
            np.savez('saved_models/encoder_tmp'+str(epoch)+"_"+str(test_loss/test_batches)+'.npz', *lasagne.layers.get_all_param_values(bottleneck))
        
        ## sometimes save/show bottleneck activation, filters...
        show = False
        if show and epoch>0 and epoch%5==0: 
            
            print("saving filters..."); utils.get_filters(network, epoch)
        
            ### show bottleneck activation of one random test spectrogram
            i = np.random.randint(0, test_data.shape[0])
            t = test_data[i:i+1]
            bn = bottleneck_activation_fn(t)
            out = output_fn(t)
            ##print(bn.shape)
            #print("DBG output max: "+str(out.max()))
            
            #fig, axes = plt.subplots(bn.shape[1]/2, 2) # 32 filters
            #for i, ax in enumerate(axes.ravel()): 
                #ax.pcolormesh(bn[0][i])
            #plt.show()
            
            for i in range(1):
                f, axis = plt.subplots(3)
                axis[0].pcolormesh(bn[0][i])
                axis[0].set_title("bottleneck activation")
                axis[1].pcolormesh(t[0][0])     
                axis[1].set_title("input")
                axis[2].pcolormesh(out[0][0])
                axis[2].set_title("output")
                plt.show()
            
            
    
    ## save encoding network:
    print("saving final encoding network...")
    np.savez('encoder.npz', *lasagne.layers.get_all_param_values(bottleneck))
    
    #print("saving reconstruction examples of test set...")
    #output = output_fn(test_data[:2])
    #np.save('out.npy', np.array([output[:,0], test_data[:,0]])) # [0][0]: first index of example, second of (unique) channel
    
    bn_output = bottleneck_activation_fn(test_data)
    print("bottleneck activation size: "+str(bn_output.shape))
    # print(bn_output[0])
    
    
    return best_validation_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains the autoencoder.')
    parser.add_argument('-d', '--data_path', 
                        help='the path of the dataset',
                        default=None, required=True)
    if len(sys.argv)>1:
        data_path = sys.argv[1]
    print(data_path)
    train(500, 0.000005, batch_size=256)
    
