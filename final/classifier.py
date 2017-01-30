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
    target_mean_var = T.vector('target_mean')
    print("building the classifier...")
    network, bottleneck = utils.build_classifier(classifier_path=classifier_path, input_data=train_data, 
                                                    input_labels=train_labels, input_var=input_var)
    
    prediction = lasagne.layers.get_output(network, deterministic=True)
    mean_prediction = lasagne.layers.get_output(network, deterministic=True).mean(axis=0)
    
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var).mean()
    
    accuracy = T.mean(T.eq(T.argmax(prediction, axis=1), T.argmax(target_var, axis=1)), dtype=theano.config.floatX)
    mean_outcome = T.eq(T.argmax(mean_prediction), T.argmax(target_mean_var))     # 1 if the mean prediction is correct, 0 otherwise...
    
    test_fn = theano.function([input_var, target_var, target_mean_var], [loss, accuracy, mean_outcome])
    
    test_loss = 0
    test_accuracy = 0
    test_outcome = 0
    test_batches = 0
    for data, labels in zip(test_data, test_labels):
        tmp_loss, tmp_acc, tmp_outcome = test_fn(data, labels, labels[0])
        test_loss += tmp_loss
        test_accuracy += tmp_acc
        test_outcome += tmp_outcome
        test_batches += 1.0

    print("test loss = "+str(test_loss/test_batches)+", test accuracy = "+str(test_accuracy/test_batches)+", average accuracy = "+str(test_outcome/test_batches))
    

def train(num_epochs, LR, data_path, batch_size=256, encoder_path=None, output_path=None):
    train_data, train_labels, val_data, val_labels, test_data, test_labels = utils.load_dataset(data_path)
    input_var = T.tensor4('inputs')
    target_var = T.matrix('targets')
    target_mean_var = T.vector('target_mean')
    print("building the classifier...")
    network, bottleneck = utils.build_classifier(encoder_path = encoder_path, input_data=train_data, input_labels=train_labels, input_var=input_var)
    
    prediction = lasagne.layers.get_output(network)
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    mean_prediction = test_prediction.mean(axis=0)
    
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
    test_loss = test_loss.mean()
   
    test_accuracy = T.mean(T.eq(T.argmax(test_prediction, axis=1), T.argmax(target_var, axis=1)),
                        dtype=theano.config.floatX)
    mean_outcome = T.eq(T.argmax(mean_prediction), T.argmax(target_mean_var))     # 1 if the mean prediction is correct, 0 otherwise...
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.adam(loss, params, learning_rate=LR)
    
    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    output_fn = theano.function([input_var], test_prediction)
    val_fn = theano.function([input_var, target_var], [test_loss, test_accuracy])
    test_fn = theano.function([input_var, target_var, target_mean_var], [test_loss, test_accuracy, mean_outcome])
    
    best_validation_loss = 10
    if output_path is not None:
        progress = open("progress_"+output_path+".txt", "w")
    print("epoch,   validation,           test")
    for epoch in range(num_epochs):

        train_loss = []
        for batch_data, batch_labels in utils.iterate_minibatches(train_data, train_labels, batch_size):
            tmp = train_fn(batch_data, batch_labels)
            train_loss.append(tmp)
        
        val_loss = []
        val_accuracy = []
        for batch_data, batch_labels in utils.iterate_minibatches(val_data, val_labels, 10):
            tmp_loss, tmp_acc = val_fn(batch_data, batch_labels)
            val_loss.append(tmp_loss)
            val_accuracy.append(tmp_acc)
        
        test_loss = []
        test_accuracy = []
        test_outcome = []
        for data, labels in zip(test_data, test_labels):
            tmp_loss, tmp_acc, tmp_outcome = test_fn(data, labels, labels[0])
            #~ out = output_fn(data)
            #~ print(out.max(axis=1))
            test_loss.append(tmp_loss)
            test_accuracy.append(tmp_acc)
            test_outcome.append(tmp_outcome)
            
        val_loss = np.array(val_loss).mean()
        val_accuracy = np.array(val_accuracy).mean()*100
        test_loss = np.array(test_loss).mean()
        test_accuracy = np.array(test_accuracy).mean()*100
        test_mean_accuracy = np.array(test_outcome).mean()*100
        
        perf_line = (str(epoch)+",    "+str(val_loss)+", "+"{:.2f}".format(val_accuracy)+",    "+str(test_loss)+", "+"{:.2f}".format(test_accuracy)+", "+"{:.2f}".format(test_mean_accuracy))
        
        print(perf_line)
        if output_path is not None: 
            progress = open("progress_"+output_path+".txt", "a", 0)
            progress.write(perf_line+"\n")
            progress.close()
        
        if best_validation_loss > val_loss:
            best_validation_loss = val_loss
            if output_path is not None and best_validation_loss < 1:
                print("saving classifier..."); np.savez('classifier'+str(output_path)+'.npz', *lasagne.layers.get_all_param_values(network))
                print("saving filters..."); utils.get_filters(network, output_path)
            
        
    
if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description='Trains or tests the classifier.')
    parser.add_argument('mode', 
                        help='\'train\' or \'test\'')
    parser.add_argument('-d', '--data_path', 
                        help='the path of the dataset, for both training and testing',
                        default=None, required=True)
    parser.add_argument('-p', '--parameters_path', 
                        help='the path of the saved encoder in \'train\' mode (optional), or the path of the saved classifier in \'test\' mode', 
                        required=False)
    parser.add_argument('-o', '--output_path', 
                        help='the path to the output classifier and progress report',
                        default=None, required=False)
    args = parser.parse_args()
    
    print("data: "+str(args.data_path))
    if "train" in args.mode:
        train(500, 0.00007, args.data_path, batch_size=256, encoder_path=args.parameters_path, output_path=args.output_path) # better with 0.00007
    elif "test" in args.mode:
        use(args.data_path, classifier_path=args.parameters_path)

