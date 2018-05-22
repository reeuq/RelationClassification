from gensim.models import word2vec
import numpy as np
from six.moves import cPickle as pickle


if __name__ == "__main__":
    model = word2vec.KeyedVectors.load_word2vec_format('./../resource/original/GoogleNews-vectors-negative300.bin', binary=True)

    #加载所有的句子
    pickle_file = './../resource/newSentence.pickle'
    with open(pickle_file, 'rb') as f:
        save = pickle.load(f)
        sentences = save['sentences']
        sentences_len = save['sentencesLen']
        labels = save['labels']
        entityPairs = save['entityPairs']
        entityPairs_len = save['entityPairsLen']
        del save

    # pickle_file = './../resource/test_sentence.pickle'
    # with open(pickle_file, 'rb') as f:
    #     save = pickle.load(f)
    #     test_sentences = save['test_sentences']
    #     del save
    #     print('sentences', test_sentences)

    temp = np.array(sentences_len)
    sentences_id_len = np.sum(temp, axis=1)
    entityPairs_len = np.array(entityPairs_len)
    entityPairs_max_len = np.max(entityPairs_len)

    cut1 = np.delete(temp, [3, 4], axis=1)
    bias1 = np.max(np.sum(cut1, axis=1))
    cut2 = np.delete(temp, [0, 1], axis=1)
    bias2 = np.max(np.sum(cut2, axis=1))
    pos_dict_len = bias1 + bias2 + 1  #1表示“0”这个位置，-3、-2、-1、0、0、1、2、3、4 共为3+4+1

    pos_vec1 = []
    for each_len in sentences_len:
        pos = []
        for i in range(each_len[0], 0, -1):
            pos.append(i)
        for i in range(each_len[1], 0, -1):
            pos.append(0)
        for i in range(1, each_len[2] + each_len[3] + each_len[4] + 1):
            pos.append(bias1 + i)
        pos_vec1.append(pos)

    pos_vec2 = []
    for each_len in sentences_len:
        pos = []
        for i in range(each_len[0] + each_len[1] + each_len[2], 0, -1):
            pos.append(i)
        for i in range(each_len[3], 0, -1):
            pos.append(0)
        for i in range(1, each_len[4] + 1):
            pos.append(bias1 + i)
        pos_vec2.append(pos)

    #获取所有的词汇
    words = set()
    for sentence in sentences:
        words.update(sentence.split())

    # for sentence in test_sentences:
    #     words.update(sentence.split())

    #组成词典,并获取embedding层的W初始值
    dictionary = dict()
    wordEmbedding = [np.zeros(300, dtype=np.float32)]
    for i, word in enumerate(words):
        dictionary[word] = i + 1
        try:
            wordVec = model.wv[word]
            wordEmbedding.append(wordVec)
        except Exception:
            wordEmbedding.append(-1 + 2 * np.random.random_sample(300))

    #获取句子向量序列号
    train_max_len = 0
    sentences_vec = []
    for sentence in sentences:
        sentence_vec = []
        for word in sentence.split():
            sentence_vec.append(dictionary.get(word))
        sentences_vec.append(sentence_vec)
        if len(sentence.split()) > train_max_len:
            train_max_len = len(sentence.split())

    entityPairs_vec = []
    for entityPair in entityPairs:
        entityPair_vec = []
        for word in entityPair.split():
            entityPair_vec.append(dictionary.get(word))
        entityPairs_vec.append(entityPair_vec)

    # test_len = 0
    # test_sentences_vec = []
    # for sentence in test_sentences:
    #     sentence = sentence.translate(str.maketrans('/-', '  ', punc))
    #     sentence_vec = []
    #     for word in sentence.split():
    #         sentence_vec.append(dictionary.get(word))
    #     test_sentences_vec.append(sentence_vec)
    #     if sentence.split().__len__() > test_len:
    #         test_len = sentence.split().__len__()

    with open('./../resource/newDictionary.pickle', 'wb') as f:
        save = {
            'dictionary': dictionary,
            'wordEmbedding': wordEmbedding,
            'sentences_vec': sentences_vec,
            'sentences_id_len': sentences_id_len,
            'pos_vec1': pos_vec1,
            'pos_vec2': pos_vec2,
            'entityPairs_vec': entityPairs_vec,
            'entityPairs_len': entityPairs_len,
            'labels': labels
            # 'test_sentences_vec': test_sentences_vec
        }
        pickle.dump(save, f, protocol=2)
    print("train len ", train_max_len)
    print('train pos dic len', pos_dict_len)
    print("train entity len ", entityPairs_max_len)
    # print("test len ", test_len)