import numpy as np
import lasagne

#####


##### dataset loaders
def load_dataset(data_path):
    spectrograms = np.load(data_path)
    spectrograms = spectrograms[:,:10,:] # example, frequency, time
    data = np.empty((spectrograms.shape[0], 1, spectrograms.shape[1], spectrograms.shape[2]))
    for i in range(spectrograms.shape[0]):
            data[i,0,:,:] = spectrograms[i].reshape(1,spectrograms.shape[1],spectrograms.shape[2])
    print(data.shape)
    percent_train_data = 0.7
    n_train_examples = int(percent_train_data*data.shape[0])
    train_data = data[:n_train_examples].astype(np.float32)
    test_data = data[n_train_examples:].astype(np.float32)
    print(train_data.shape)
    print(test_data.shape)
    return train_data, test_data

## loads the dataset from a npz archive, returns a single (merged) training array and a list of test arrays
def load_dataset_zipped(data_path, train_size=.70, network_type = 'convolutional'):    
    spectrograms = np.load(data_path)
    spectrograms = [spectrograms[f] for f in spectrograms.files]
    data = []
    # introducing (dummy) channel axis
    if 'fully connected' in network_type:
        for sp in spectrograms:
            tmp = [np.ravel(s) for s in sp]
            data.append(np.array(tmp))
    else:
        for sp in spectrograms:
            tmp = np.empty((sp.shape[0], 1, sp.shape[1], sp.shape[2]))
            for i in range(sp.shape[0]):
                    tmp[i,0,:,:] = sp[i].reshape(1,sp.shape[1],sp.shape[2])
            #print(tmp.shape)
            data.append(tmp.astype(np.float32))
    
    n_train_examples = int(train_size*len(data))
    train_data = data[:n_train_examples]
    train_data = np.concatenate(train_data)
    test_data = data[n_train_examples:]
    print("data shapes:")
    print(train_data.shape)
    print(len(test_data))
    return train_data, test_data

## loads the dataset from a npz archive, returns training and test examples and labels
def load_dataset_zipped_supervised(data_path, train_size=.70, network_type = 'convolutional', shuffle=False):    ##
    spectrograms = np.load(data_path)
    spectrograms = [spectrograms[f] for f in spectrograms.files]
    data = []
    labels = []
    # introducing (dummy) channel axis
    
    for (n, sp) in enumerate(spectrograms): # for each song
        tmp = np.empty((sp.shape[0], 1, sp.shape[1], sp.shape[2]))
        labels_song = []
        for i in range(sp.shape[0]):    # for each spectrogram (i.e. bit of the song of sp)
            one_hot_label = np.zeros(len(spectrograms), dtype=np.int); one_hot_label[n]=1
            labels_song.append(one_hot_label)
            tmp[i,0,:,:] = sp[i].reshape(1,sp.shape[1],sp.shape[2])
        labels.append(np.array(labels_song).astype(np.float32))
        data.append(tmp.astype(np.float32))
    
    if shuffle:
        data = np.concatenate(data)
        n_train_examples = int(train_size*data.shape[0])
        labels = np.concatenate(labels)
        indices = np.arange(data.shape[0])
        np.random.shuffle(indices)
        return data[indices[:n_train_examples]], labels[indices[:n_train_examples]], data[indices[n_train_examples:]], labels[indices[n_train_examples:]]
        
    n_train_examples = int(train_size*len(data))
    train_data = data[:n_train_examples]
    train_labels = labels[:n_train_examples]
    test_data = data[n_train_examples:]
    test_labels = labels[n_train_examples:]
    train_data = np.concatenate(train_data)
    train_labels = np.concatenate(train_labels)
    test_data = np.concatenate(test_data)
    test_labels = np.concatenate(test_labels)
    print("data shapes:")
    print("number of songs for training = "+str(n_train_examples))
    print(train_data.shape)
    print(train_labels.shape)
    return train_data, train_labels, test_data, test_labels
    
    

def build_network_convolutional(input_data=None, input_var=None, n_filters=64, filter_size=(3,5)):
    
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
    input_layer = lasagne.layers.InputLayer((None, 1, freq_size, time_size), input_var=input_var)
    
    ##### encoder
    en_conv1 = lasagne.layers.Conv2DLayer(
            input_layer, num_filters = n_filters, filter_size=filter_size,
            #~ stride = (2,2),
            pad = 'same',
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    
    en_pool1 = lasagne.layers.MaxPool2DLayer(en_conv1, (2,2))
    
    en_conv2 = lasagne.layers.Conv2DLayer(
            en_pool1, num_filters = n_filters, filter_size=filter_size,
            #~ stride = (2,2),
            pad = 'same',
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    bottleneck = en_conv2

    print("encoder ok")
    
    ##### decoder
    
    de_conv2 = lasagne.layers.Conv2DLayer(
            en_conv2, num_filters = n_filters, filter_size=filter_size,
            #~ stride = (2,2),
            pad = 'same',
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    
    #de_pool2 = lasagne.layers.Upscale2DLayer(de_conv2, (2,1))
    de_pool2 = lasagne.layers.InverseLayer(de_conv2, en_pool1)
    
    de_conv3 = lasagne.layers.Conv2DLayer(
            de_pool2, num_filters = 1, filter_size=filter_size,
            #~ stride = (2,2),
            pad = 'same',
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    network = de_conv3
    
    print("decoder ok")
    return network, bottleneck
    
    
    #build the classifier with another fully connected layer
def build_classifier(input_data=None, input_labels=None, input_var=None, n_filters=64, filter_size=(3,5)):
    #num_classes = max(input_labels)+1
    num_classes = input_labels.shape[1]
    print(num_classes)
    network, bottleneck = build_network_convolutional(input_data=input_data, input_var=input_var, n_filters=n_filters, filter_size=filter_size)
    print("loading encoder parameters...")
    with np.load('encoder.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(bottleneck, param_values)
    
    for layer in lasagne.layers.get_all_layers(bottleneck):
        for param in layer.params:
            layer.params[param].discard('trainable')
    
    cl_dense0 = lasagne.layers.DenseLayer(bottleneck, int(num_classes*4+80), nonlinearity=lasagne.nonlinearities.tanh)
    cl_dense1 = lasagne.layers.DenseLayer(cl_dense0, int(num_classes*2+40), nonlinearity=lasagne.nonlinearities.tanh)
    #~ cl_dense2 = lasagne.layers.DenseLayer(cl_dense1, int(num_classes*2), nonlinearity=lasagne.nonlinearities.tanh)
    cl_out = lasagne.layers.DenseLayer(cl_dense1, num_classes, nonlinearity=lasagne.nonlinearities.softmax)
    #~ cl_out = lasagne.layers.DenseLayer(cl_dense2, num_classes, nonlinearity=None) # linear layer, we applay logsoftmax after
    network = cl_out
    
    return network


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]
    
    
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
