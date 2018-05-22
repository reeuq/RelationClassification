from six.moves import cPickle as pickle


with open('./../resource/newSentence.pickle', 'rb') as f:
    save = pickle.load(f)
    pass