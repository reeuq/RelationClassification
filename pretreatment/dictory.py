from gensim.models import word2vec
import string
import numpy as np
from six.moves import cPickle as pickle


if __name__ == "__main__":
    model = word2vec.KeyedVectors.load_word2vec_format('./../resource/original/GoogleNews-vectors-negative300.bin', binary=True)
    punc = string.punctuation.replace('-', '').replace('/', '')

    #加载所有的句子
    pickle_file = './../resource/sentence.pickle'
    with open(pickle_file, 'rb') as f:
        save = pickle.load(f)
        sentences = save['sentences']
        del save
        print('sentences', sentences)

    pickle_file = './../resource/test_sentence.pickle'
    with open(pickle_file, 'rb') as f:
        save = pickle.load(f)
        test_sentences = save['test_sentences']
        del save
        print('sentences', test_sentences)

    #获取所有的词汇
    words = set()
    for sentence in sentences:
        sentence = sentence.translate(str.maketrans('/-', '  ', punc))
        words.update(sentence.split())

    for sentence in test_sentences:
        sentence = sentence.translate(str.maketrans('/-', '  ', punc))
        words.update(sentence.split())

    #组成词典,并获取embedding层的W初始值
    dictionary = {}
    W = [np.zeros(300, dtype=np.float32)]
    for i, word in enumerate(words):
        dictionary[word] = i + 1
        try:
            wordVec = model.wv[word]
            W.append(wordVec)
        except Exception:
            W.append(-1 + 2 * np.random.random_sample(300))

    #获取句子向量序列号
    train_len = 0
    sentences_vec = []
    for sentence in sentences:
        sentence = sentence.translate(str.maketrans('/-', '  ', punc))
        sentence_vec = []
        for word in sentence.split():
            sentence_vec.append(dictionary.get(word))
        sentences_vec.append(sentence_vec)
        if sentence.split().__len__() > train_len:
            train_len = sentence.split().__len__()

    test_len = 0
    test_sentences_vec = []
    for sentence in test_sentences:
        sentence = sentence.translate(str.maketrans('/-', '  ', punc))
        sentence_vec = []
        for word in sentence.split():
            sentence_vec.append(dictionary.get(word))
        test_sentences_vec.append(sentence_vec)
        if sentence.split().__len__() > test_len:
            test_len = sentence.split().__len__()

    with open('./../resource/dictionary.pickle', 'wb') as f:
        save = {
            'dictionary': dictionary,
            'W': W,
            'sentences_vec': sentences_vec,
            'test_sentences_vec': test_sentences_vec
        }
        pickle.dump(save, f, protocol=2)
    print("train len ", train_len)
    print("test len ", test_len)