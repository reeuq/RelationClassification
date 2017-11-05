from six.moves import cPickle as pickle

with open('./resource/vec.pickle', 'rb') as f:
    save = pickle.load(f)
    dataset = save['dataset']
    label = save['label']
    del save
    print('dataset', dataset.shape)
    print('label', label.shape)