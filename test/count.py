from six.moves import cPickle as pickle

with open('./../resource/vec.pickle', 'rb') as f:
    save = pickle.load(f)
    dataset = save['dataset']
    label = save['label']
    del save
    count = [0]*11
    for a in label:
        if a == 0:
            count[0] += 1
        elif a == 1:
            count[1] += 1
        elif a == 2:
            count[2] += 1
        elif a == 3:
            count[3] += 1
        elif a == 4:
            count[4] += 1
        elif a == 5:
            count[5] += 1
        elif a == 6:
            count[6] += 1
        elif a == 7:
            count[7] += 1
        elif a == 8:
            count[8] += 1
        elif a == 9:
            count[9] += 1
        elif a == 10:
            count[10] += 1
    print(count)
    print('dataset', dataset.shape)
    print('label', label.shape)