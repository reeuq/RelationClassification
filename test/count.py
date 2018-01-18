from six.moves import cPickle as pickle

with open('./../resource/vec.pickle', 'rb') as f:
    save = pickle.load(f)
    label = save['label']
    del save
    with open('./../resource/sentence.pickle', 'rb') as file:
        save = pickle.load(file)
        sentences = save['sentences']
        del save
        sentenceFile = ['']*11
        for i in range(sentences.__len__()):
            if label[i] == 0:
                sentenceFile[0] += (sentences[i] + '\n')
            elif label[i] == 1:
                sentenceFile[1] += (sentences[i] + '\n')
            elif label[i] == 2:
                sentenceFile[2] += (sentences[i] + '\n')
            elif label[i] == 3:
                sentenceFile[3] += (sentences[i] + '\n')
            elif label[i] == 4:
                sentenceFile[4] += (sentences[i] + '\n')
            elif label[i] == 5:
                sentenceFile[5] += (sentences[i] + '\n')
            elif label[i] == 6:
                sentenceFile[6] += (sentences[i] + '\n')
            elif label[i] == 7:
                sentenceFile[7] += (sentences[i] + '\n')
            elif label[i] == 8:
                sentenceFile[8] += (sentences[i] + '\n')
            elif label[i] == 9:
                sentenceFile[9] += (sentences[i] + '\n')
            elif label[i] == 10:
                sentenceFile[10] += (sentences[i] + '\n')

        with open('./../resource/sentence.txt', 'w') as f:
            for i in range(sentenceFile.__len__()):
                f.write(str(sentenceFile[i]) + '\n')