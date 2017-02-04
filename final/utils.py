import numpy as np
import lasagne

#####


##### dataset loaders

def load_dataset(data_path):
    print("loading dataset...")
    spectrograms = np.load(data_path+"_clean.npz")
    # add dummy axis for single channel
    tr_data = spectrograms['arr_0']
    tr_labels = spectrograms['arr_1'].astype(np.float32)
    val_data = spectrograms['arr_2']
    val_labels = spectrograms['arr_3'].astype(np.float32)
    
    spectrograms = np.load(data_path+"_distorted.npz")
    test_data = spectrograms['arr_0']
    test_labels = spectrograms['arr_1'].astype(np.float32)
    tr_data = np.expand_dims(tr_data, axis=1)
    val_data = np.expand_dims(val_data, axis=1)
    test_data = np.expand_dims(test_data, axis=2) # 2nd axis because it's bagged
    print(tr_data.shape)
    print(tr_labels.shape)
    print(val_data.shape)
    print(val_labels.shape)
    print(test_data.shape)
    print(test_labels.shape)
    return tr_data, tr_labels, val_data, val_labels, test_data, test_labels

## loads the dataset from a npz archive, returns a single (merged) training array and a list of test arrays
#~ def load_dataset_zipped(data_path, train_size=.70):    
    #~ spectrograms = np.load(data_path)
    #~ spectrograms = [spectrograms[f] for f in spectrograms.files]
    #~ data = []
    #~ # introducing (dummy) channel axis
    #~ if 'fully connected' in network_type:
        #~ for sp in spectrograms:
            #~ tmp = [np.ravel(s) for s in sp]
            #~ data.append(np.array(tmp))
    #~ else:
        #~ for sp in spectrograms:
            #~ tmp = np.empty((sp.shape[0], 1, sp.shape[1], sp.shape[2]))
            #~ for i in range(sp.shape[0]):
                    #~ tmp[i,0,:,:] = sp[i].reshape(1,sp.shape[1],sp.shape[2])
            #~ data.append(tmp.astype(np.float32))
    
    #~ n_train_examples = int(train_size*len(data))
    #~ train_data = data[:n_train_examples]
    #~ train_data = np.concatenate(train_data)
    #~ test_data = data[n_train_examples:]
    #~ print("data shapes:")
    #~ print("  "+str(train_data.shape))
    #~ print("  "+str(len(test_data)))
    #~ return train_data, test_data
    

#~ ## loads the dataset from a npz archive, returns training and test examples and labels
#~ def load_dataset_zipped_supervised(data_path, train_size=.70, bag_size = 5):    ##
    #~ spectrograms = np.load(data_path)
    #~ spectrograms = [spectrograms[f] for f in spectrograms.files]
    #~ data = []
    #~ labels = []
    #~ bagged_data = []
    #~ bagged_labels = []
    #~ # introducing (dummy) channel axis
    
    #~ for (n, sp) in enumerate(spectrograms): # for each song
        #~ tmp = np.empty((sp.shape[0], 1, sp.shape[1], sp.shape[2]))
        #~ labels_song = []
        #~ for i in range(sp.shape[0]):    # for each spectrogram (i.e. bit of the song of sp)
            #~ one_hot_label = np.zeros(len(spectrograms), dtype=np.int); one_hot_label[n]=1
            #~ labels_song.append(one_hot_label)
            #~ tmp[i,0,:,:] = sp[i].reshape(1,sp.shape[1],sp.shape[2])
        #~ labels.append(np.array(labels_song).astype(np.float32))
        #~ data.append(tmp.astype(np.float32))
        
    #~ ## data for bagged classification:
    #~ # we want a list of numpy arrays, each array contains bag_size equal sized spectrograms. the final prediction is the average of the predictions of these spectrograms
    #~ for (n, sp) in enumerate(spectrograms): # for each song
        #~ #print("DBG: song shape  "+str(sp.shape))
        #~ bag_ids = np.array_split(np.arange(len(sp)), int(len(sp)/bag_size)) # split the indices of the spectrograms in subarrays
        #~ for ids in bag_ids:    # for each bag of spectrograms
            #~ tmp = np.empty((len(ids), 1, sp.shape[1], sp.shape[2]))
            #~ labels_bag = []
            #~ one_hot_label = np.zeros(len(spectrograms), dtype=np.int); one_hot_label[n]=1
            #~ for i, id in enumerate(ids):
                #~ tmp[i,0,:,:] = sp[id].reshape(1,sp.shape[1],sp.shape[2])
                #~ labels_bag.append(one_hot_label)
            #~ bagged_labels.append(np.array(labels_bag).astype(np.float32))
            #~ bagged_data.append(tmp.astype(np.float32))
    
    #~ data = np.concatenate(data)
    #~ labels = np.concatenate(labels)
    #~ n_train_examples = int(train_size*data.shape[0])
    #~ indices = np.arange(data.shape[0])
    #~ np.random.shuffle(indices)
    
    #~ return data[indices[:n_train_examples]], labels[indices[:n_train_examples]], data[indices[n_train_examples:]], labels[indices[n_train_examples:]], bagged_data, bagged_labels
    
def get_filters(network, id=""):
    import matplotlib.pyplot as plt
    params = lasagne.layers.get_all_layers(network)[2].get_params()[0].get_value()  # the first index is the index of the layer... 2 is because there is a dropout in between input and 1st hidden layer
    for f in range(params.shape[0]):
        params[f,:,:,:] -= params[f,:,:,:].min()
        params[f,:,:,:] /= params[f,:,:,:].max()
    print("filter shape "+str(params.shape))
    
    filters = []
    for i,p in enumerate(params):
        #~ plt.imshow(p[0,:,:], cmap='gray', interpolation='none')
        #~ plt.savefig("filters/filter_"+str(id)+"_"+str(i)+".pdf", bbox_inches='tight')
        filters.append(p)
    np.save("filters/filters_"+str(id), np.array(filters))
        

    
n_filters=64
filter_size=(7,15)

def build_autoencoder(input_data=None, input_var=None, input_dropout_p = 0.3):
    print(str(n_filters)+" filters "+str(filter_size)+",  "+str(input_dropout_p)+" input dropout")
    #####

    freq_size = input_data.shape[2]
    time_size = input_data.shape[3]
    input_layer = lasagne.layers.InputLayer((None, 1, freq_size, time_size), input_var=input_var)
    
    ##### encoder
    en_conv1 = lasagne.layers.Conv2DLayer(
            lasagne.layers.DropoutLayer(input_layer, p=input_dropout_p), 
            num_filters = n_filters, filter_size=filter_size,
            pad = 'same',
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    
    en_pool1 = lasagne.layers.MaxPool2DLayer(en_conv1, (2,2))
    
    en_conv2 = lasagne.layers.Conv2DLayer(
            en_pool1, 
            num_filters = n_filters, filter_size=filter_size,
            pad = 'same',
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    
    

    bottleneck = en_conv2

    print("encoder ok")
    
    #~ ##### decoder
    
    
    de_conv2 = lasagne.layers.Conv2DLayer(
            en_conv2, 
            num_filters = n_filters, filter_size=filter_size,
            pad = 'same',
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    
    de_pool2 = lasagne.layers.InverseLayer(de_conv2, en_pool1)
    
    de_conv3 = lasagne.layers.Conv2DLayer(
            de_pool2, 
            num_filters = 1, filter_size=filter_size,
            pad = 'same',
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    network = de_conv3
    
    print("decoder ok")
    return network, bottleneck
    
    
    #build the classifier with another fully connected layer
def build_classifier(encoder_path = None, input_data=None, input_labels=None, input_var=None, classifier_path=None):
    num_classes = input_labels.shape[1] # compute num of classes based on the size of the one-hot encoded labels
    print("Number of classes: "+str(num_classes))
    
    # rebuild the encoder and load the trained parameters
    network, bottleneck = build_autoencoder(input_data=input_data, input_var=input_var, input_dropout_p = 0.2)
    if encoder_path is None:
        if classifier_path is None: print("using untrained encoder")
    else:
        print("loading encoder parameters from "+encoder_path)
        with np.load(encoder_path) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(bottleneck, param_values)
    
    
    ## freeze encoder parameters
    #print("freezing encoder parameters!")
    #for layer in lasagne.layers.get_all_layers(bottleneck):
        #for param in layer.params:
            #layer.params[param].discard('trainable')
    
    #~ cl_dense0 = lasagne.layers.DenseLayer(bottleneck, int(num_classes*2), nonlinearity=lasagne.nonlinearities.tanh)
    #~ cl_dense1 = lasagne.layers.DenseLayer(lasagne.layers.DropoutLayer(cl_dense0, p=0.2), 
                                        #~ int(num_classes*1.3), nonlinearity=lasagne.nonlinearities.tanh)
    cl_dense0 = lasagne.layers.DenseLayer(bottleneck, int(num_classes*2), nonlinearity=lasagne.nonlinearities.tanh)
    cl_dense1 = lasagne.layers.DenseLayer(lasagne.layers.DropoutLayer(cl_dense0, p=0.2), 
                                        int(num_classes*1.3), nonlinearity=lasagne.nonlinearities.tanh)
    cl_dense2 = lasagne.layers.DenseLayer(lasagne.layers.DropoutLayer(cl_dense0, p=0.2), 
                                        int(num_classes*1.1), nonlinearity=lasagne.nonlinearities.tanh)
    
    cl_out = lasagne.layers.DenseLayer(cl_dense2, num_classes, nonlinearity=lasagne.nonlinearities.softmax)
    #~ cl_out = lasagne.layers.DenseLayer(cl_dense2, num_classes, nonlinearity=None) # linear layer, we applay logsoftmax after
    network = cl_out
    
    if classifier_path is not None:
        print("loading classifier parameters...")
        with np.load(classifier_path) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(network, param_values)
        
    return network, bottleneck


def build_dummy_classifier(input_data=None, input_labels=None, input_var=None, classifier_path=None):
    num_classes = input_labels.shape[1] # compute num of classes based on the size of the one-hot encoded labels
    print("Number of classes: "+str(num_classes))
    
    freq_size = input_data.shape[2]
    time_size = input_data.shape[3]
    input_layer = lasagne.layers.InputLayer((None, 1, freq_size, time_size), input_var=input_var)
    bottleneck = network = lasagne.layers.MaxPool2DLayer(input_layer, (2,2))
    
    cl_dense0 = lasagne.layers.DenseLayer(bottleneck, int(num_classes*2), nonlinearity=lasagne.nonlinearities.tanh)
    cl_dense1 = lasagne.layers.DenseLayer(lasagne.layers.DropoutLayer(cl_dense0, p=0.2), 
                                        int(num_classes*1.3), nonlinearity=lasagne.nonlinearities.tanh)
    cl_dense2 = lasagne.layers.DenseLayer(lasagne.layers.DropoutLayer(cl_dense0, p=0.2), 
                                        int(num_classes*1.1), nonlinearity=lasagne.nonlinearities.tanh)
    
    cl_out = lasagne.layers.DenseLayer(cl_dense2, num_classes, nonlinearity=lasagne.nonlinearities.softmax)
    #~ cl_out = lasagne.layers.DenseLayer(cl_dense2, num_classes, nonlinearity=None) # linear layer, we applay logsoftmax after
    network = cl_out
    
    if classifier_path is not None:
        print("loading classifier parameters...")
        with np.load(classifier_path) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(network, param_values)
        
    return network, bottleneck


def iterate_minibatches(inputs, targets, batchsize, shuffle=True):
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
