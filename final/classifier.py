import numpy as np
import os
import sys
import time

import theano
import theano.tensor as T
import lasagne
import utils

data_path = ""

### TODO: try binary classifier!

##### functions needed not to get NaN with cross entropy
def log_softmax(x):
    xdev = x - x.max(1, keepdims=True)
    return xdev - T.log(T.sum(T.exp(xdev), axis=1, keepdims=True))

def categorical_crossentropy_logdomain(log_predictions, targets):
    return -T.sum(targets * log_predictions, axis=1)
    
    
    

def train(num_epochs, LR, batch_size=256, encoder_path=None):
    train_data, train_labels, test_data, test_labels, bagged_data, bagged_labels = utils.load_dataset_zipped_supervised(data_path, train_size = .75)
    input_var = T.tensor4('inputs')
    target_var = T.matrix('targets')
    print("building the classifier...")
    network, bottleneck = utils.build_classifier(encoder_path = encoder_path, input_data=train_data, input_labels=train_labels, input_var=input_var)
    
    prediction = lasagne.layers.get_output(network)
    #~ prediction = log_softmax(lasagne.layers.get_output(network))
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    #~ loss = categorical_crossentropy_logdomain(prediction, target_var)
    loss = loss.mean()
    accuracy = T.mean(T.eq(T.argmax(prediction, axis=1), T.argmax(target_var, axis=1)),
                        dtype=theano.config.floatX)
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.adam(loss, params, learning_rate=LR)
    
    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    output_fn = theano.function([input_var], prediction)
    val_fn = theano.function([input_var, target_var], [loss, accuracy])
    
    best_validation_loss = 10
    for epoch in range(num_epochs):
        ## dbg logsoftmax
        #~ out = output_fn(train_data[5:6])
        #~ print(out)
        ####
        train_loss = 0
        train_batches = 0
        for batch_data, batch_labels in utils.iterate_minibatches(train_data, train_labels, batch_size):
            tmp = train_fn(batch_data, batch_labels)
            #~ print(tmp)
            train_loss += tmp
            train_batches += 1
            #~ print("Batch "+str(train_batches))
        #print("Epoch "+str(epoch)+", training loss = "+str(train_loss/train_batches))
        
        test_loss = 0
        test_accuracy = 0
        test_batches = 0
        for batch_data, batch_labels in utils.iterate_minibatches(test_data, test_labels, 10):
            tmp_loss, tmp_acc = val_fn(batch_data, batch_labels)
            #~ print(tmp)
            test_loss += tmp_loss
            test_accuracy += tmp_acc
            test_batches += 1
            #~ print("Batch "+str(train_batches))
            
        test_loss = test_loss/test_batches
        if best_validation_loss > test_loss:
            best_validation_loss = test_loss
        print("Epoch "+str(epoch)+", test loss = "+str(test_loss/test_batches)+"    test accuracy = "+str(test_accuracy/test_batches))
    
if __name__ == '__main__':
    # usage: python classifier.py <train, use> data_path encoder_path(optional)
    mode = sys.argv[1]
    data_path = sys.argv[2]
    encoder_path = None
    if len(sys.argv) > 3:
        encoder_path = sys.argv[3]
    print(data_path)
    if "train" in mode:
        train(500, 0.0005, batch_size=256, encoder_path = encoder_path)

