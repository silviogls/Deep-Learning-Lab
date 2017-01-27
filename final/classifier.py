import numpy as np
import os
import sys
import time
import argparse

import theano
import theano.tensor as T
import lasagne
import utils

### TODO: try binary classifier!

##### functions needed not to get NaN with cross entropy
def log_softmax(x):
    xdev = x - x.max(1, keepdims=True)
    return xdev - T.log(T.sum(T.exp(xdev), axis=1, keepdims=True))

def categorical_crossentropy_logdomain(log_predictions, targets):
    return -T.sum(targets * log_predictions, axis=1)
    
# this is to test the classifier on the noisy data
def use(data_path, classifier_path='classifier.npz'):
    train_data, train_labels, val_data, val_labels, test_data, test_labels = utils.load_dataset(data_path)
    input_var = T.tensor4('inputs')
    target_var = T.matrix('targets')
    print("building the classifier...")
    network, bottleneck = utils.build_classifier(classifier_path=classifier_path, input_data=train_data, 
                                                    input_labels=train_labels, input_var=input_var)
    
    prediction = lasagne.layers.get_output(network)
    #~ prediction = log_softmax(lasagne.layers.get_output(network))
    mean_prediction = lasagne.layers.get_output(network).mean(axis=0)
    
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var).mean()
    #~ loss = categorical_crossentropy_logdomain(prediction, target_var)
    
    accuracy = T.mean(T.eq(T.argmax(prediction, axis=1), T.argmax(target_var, axis=1)), dtype=theano.config.floatX)
    mean_outcome = T.eq(T.argmax(mean_prediction), T.argmax(target_var.mean(axis=0)))     # 1 if the mean prediction is correct, 0 otherwise...
    
    val_fn = theano.function([input_var, target_var], [loss, accuracy, mean_outcome])
    
    test_loss = 0
    test_accuracy = 0
    test_outcome = 0
    test_batches = 0
    for data, labels in utils.iterate_minibatches(test_data, test_labels, 100):
        tmp_loss, tmp_acc, tmp_outcome = val_fn(data, labels)
        test_loss += tmp_loss
        test_accuracy += tmp_acc
        test_outcome += tmp_outcome
        test_batches += 1.0

    print("test loss = "+str(test_loss/test_batches)+", test accuracy = "+str(test_accuracy/test_batches)+", average accuracy = "+str(test_outcome/test_batches))
    

def train(num_epochs, LR, data_path, batch_size=256, encoder_path=None, output_path=''):
    train_data, train_labels, val_data, val_labels, test_data, test_labels = utils.load_dataset(data_path)
    input_var = T.tensor4('inputs')
    target_var = T.matrix('targets')
    print("building the classifier...")
    network, bottleneck = utils.build_classifier(encoder_path = encoder_path, input_data=train_data, input_labels=train_labels, input_var=input_var)
    
    prediction = lasagne.layers.get_output(network)
    #~ prediction = log_softmax(lasagne.layers.get_output(network))
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    #~ loss = categorical_crossentropy_logdomain(prediction, target_var)
    accuracy = T.mean(T.eq(T.argmax(prediction, axis=1), T.argmax(target_var, axis=1)),
                        dtype=theano.config.floatX)
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.adam(loss, params, learning_rate=LR)
    
    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    output_fn = theano.function([input_var], prediction)
    val_fn = theano.function([input_var, target_var], [loss, accuracy])
    
    best_validation_loss = 10
    progress = open("progress_"+output_path+".txt", "a", 0)
    for epoch in range(num_epochs):

        train_loss = 0
        train_batches = 0
        for batch_data, batch_labels in utils.iterate_minibatches(train_data, train_labels, batch_size):
            tmp = train_fn(batch_data, batch_labels)
            train_loss += tmp
            train_batches += 1
        
        val_loss = 0
        val_accuracy = 0
        val_batches = 0
        for batch_data, batch_labels in utils.iterate_minibatches(val_data, val_labels, 10):
            tmp_loss, tmp_acc = val_fn(batch_data, batch_labels)
            val_loss += tmp_loss
            val_accuracy += tmp_acc
            val_batches += 1
            
        val_loss = val_loss/val_batches
        val_accuracy = val_accuracy/val_batches
        
        print("Epoch "+str(epoch)+", test loss = "+str(val_loss)+"    test accuracy = "+str(val_accuracy))#
        progress.write(str(epoch)+",  "+str(val_loss)+",    "+str(val_accuracy)+"\n")
        progress.flush()
        
        if best_validation_loss > val_loss:
            best_validation_loss = val_loss
            if best_validation_loss < 1:
                print("saving classifier..."); np.savez('classifier'+str(output_path)+'.npz', *lasagne.layers.get_all_param_values(network))
                print("saving filters..."); utils.get_filters(network, output_path)
            
        
    
if __name__ == '__main__':
    # usage: python classifier.py <train, use> data_path encoder_path(optional)
    
    parser = argparse.ArgumentParser(description='Train or use the classifier.')
    parser.add_argument('mode', 
                        help='\'train\' or \'use\'')
    parser.add_argument('-d', '--data_path', 
                        help='the path of the dataset, for both training and use',
                        default='dataset.npz', required=False)
    parser.add_argument('-p', '--parameters_path', 
                        help='the path of the saved encoder in \'train\' mode (optional), or the path of the saved classifier in \'use\' mode', 
                        required=False)
    parser.add_argument('-o', '--output_path', 
                        help='the path to the output classifier and progress report',
                        default='out_classifier', required=False)
    args = parser.parse_args()
    
    print("data: "+str(args.data_path))
    if "train" in args.mode:
        train(500, 0.00007, args.data_path, batch_size=256, encoder_path=args.parameters_path, output_path=args.output_path)
    elif "use" in args.mode:
        use(args.data_path, classifier_path=args.parameters_path)

