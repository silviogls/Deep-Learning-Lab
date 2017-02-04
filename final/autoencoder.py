import numpy as np
import os
import sys
import time

import theano
import theano.tensor as T
import lasagne
import utils
import argparse
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from PIL import Image



def train(data_path, num_epochs, LR, batch_size=256):    
    train_data, _, test_data, _, _, _ = utils.load_dataset(data_path)
    
    input_var = T.tensor4('inputs')
    target_var = T.tensor4('targets')
    
    print("Building network...")
    network, bottleneck = utils.build_autoencoder(train_data, input_var)


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
        train_loss = []
        for batch in utils.iterate_minibatches_autoencoder(train_data, batch_size):
            tmp = train_fn(batch, batch)
            train_loss.append(tmp)
        
        test_loss = []
        for batch in utils.iterate_minibatches_autoencoder(test_data, batch_size):
            tmp = val_fn(batch, batch)
            #~ print(tmp)
            test_loss.append(tmp)
        test_loss = np.array(test_loss).mean()
        
        if best_validation_loss > test_loss:
            best_validation_loss = test_loss
            #print("saving filters..."); utils.get_filters(network, epoch)
            
            ## save/show bottleneck activation, filters...
                    
                    
            if best_validation_loss < 0.04: 
                np.savez('saved_models/encoder_tmp'+str(epoch)+"_"+str(test_loss/test_batches)+'.npz', *lasagne.layers.get_all_param_values(bottleneck))
                print("backing up encoding network...")
                
        show = True
        if show: 
            ### show bottleneck activation of one random spectrogram
            i = np.random.randint(0, train_data.shape[0])
            t = train_data[i:i+1]
            bn = bottleneck_activation_fn(t)
            out = output_fn(t)
            
            print("saving activation etc...")
            for i in range(1):
                #plt.pcolormesh(bn[0][i])
                #plt.axis('off')
                ##plt.title("bottleneck activation")
                #plt.savefig("filters/bottleneck_activation.png")
                
                #plt.pcolormesh(t[0][0])     
                #plt.axis('off')
                ##plt.title("input")
                #plt.savefig("filters/input.png")
                
                #plt.pcolormesh(out[0][0])
                #plt.axis('off')
                ##plt.title("output")
                #plt.savefig("filters/output.png")
                
                f, ax = plt.subplots(3)
                ax[0].pcolormesh(t[0][i])
                ax[1].pcolormesh(out[0][i])
                ax[2].pcolormesh(bn[0][i])
                plt.savefig("filters/output"+str(epoch)+".pdf")
                plt.close()
                
        print("Epoch "+str(epoch)+", test loss = "+str(test_loss/test_batches))
        results = open("results.csv", 'a')
        results.write(str(epoch)+", "+str(test_loss/test_batches)+"\n")
        results.close()
            
        
            
    
    #print("saving reconstruction examples of test set...")
    #output = output_fn(test_data[:2])
    #np.save('out.npy', np.array([output[:,0], test_data[:,0]])) # [0][0]: first index of example, second of (unique) channel
    
    bn_output = bottleneck_activation_fn(test_data)
    print("bottleneck activation size: "+str(bn_output.shape))
    
    return best_validation_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains the autoencoder.')
    parser.add_argument('-d', '--data_path', 
                        help='the path of the dataset',
                        default=None, required=True)
    args = parser.parse_args()
    train(args.data_path, 500, 0.000005, batch_size=256)
    
