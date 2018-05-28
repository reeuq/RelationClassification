from gensim.models import word2vec
import numpy as np
from six.moves import cPickle as pickle


def get_position_id1(sen_len, bias):
    position_id = []
    for each_len in sen_len:
        pos = []
        for index in range(each_len[0], 0, -1):
            pos.append(index)
        for index in range(each_len[1], 0, -1):
            pos.append(0)
        for index in range(1, each_len[2] + each_len[3] + each_len[4] + 1):
            pos.append(bias + index)
        position_id.append(pos)
    return position_id


def get_position_id2(sen_len, bias):
    position_id = []
    for each_len in sen_len:
        pos = []
        for index in range(each_len[0] + each_len[1] + each_len[2], 0, -1):
            pos.append(index)
        for index in range(each_len[3], 0, -1):
            pos.append(0)
        for index in range(1, each_len[4] + 1):
            pos.append(bias + index)
        position_id.append(pos)
    return position_id


def get_sentence_id(sen, dic):
    sen_ids = []
    for each_sen in sen:
        sen_id = []
        for each_word in each_sen.split():
            sen_id.append(dic.get(each_word))
        sen_ids.append(sen_id)
    return sen_ids


if __name__ == "__main__":
    model = word2vec.KeyedVectors.load_word2vec_format('./../resource/original/GoogleNews-vectors-negative300.bin',
                                                       binary=True)

    # 加载所有的句子
    pickle_file = './../resource/newSentence.pickle'
    with open(pickle_file, 'rb') as f:
        train = pickle.load(f)
        sentences = train['sentences']
        sentences_len = train['sentences_len']
        labels = train['labels']
        entityPairs = train['entityPairs']
        entityPairs_len = train['entityPairs_len']
        del train
        test = pickle.load(f)
        test_sentences = test['test_sentences']
        test_sentences_len = test['test_sentences_len']
        test_labels = test['test_labels']
        test_entityPairs = test['test_entityPairs']
        test_entityPairs_len = test['test_entityPairs_len']
        del test

    sentences_id_len = np.sum(sentences_len, axis=1)
    test_sentences_id_len = np.sum(test_sentences_len, axis=1)
    sentences_max_len = max(np.max(sentences_id_len), np.max(test_sentences_id_len))

    entityPairs_id_len = np.array(entityPairs_len)
    test_entityPairs_id_len = np.array(test_entityPairs_len)
    entityPairs_max_len = max(np.max(entityPairs_id_len), np.max(test_entityPairs_id_len))

    # 相对位置特征最大长度
    cut1 = np.delete(sentences_len, [3, 4], axis=1)
    test_cut1 = np.delete(test_sentences_len, [3, 4], axis=1)
    bias1 = max(np.max(np.sum(cut1, axis=1)), np.max(np.sum(test_cut1, axis=1)))
    cut2 = np.delete(sentences_len, [0, 1], axis=1)
    test_cut2 = np.delete(test_sentences_len, [0, 1], axis=1)
    bias2 = max(np.max(np.sum(cut2, axis=1)), np.max(np.sum(test_cut2, axis=1)))
    pos_dict_len = bias1 + bias2 + 1  # 1表示“0”这个位置，-3、-2、-1、0、0、1、2、3、4 共为3+4+1

    # 获取相对位置特征向量
    position_id1 = get_position_id1(sentences_len, bias1)
    test_position_id1 = get_position_id1(test_sentences_len, bias1)
    position_id2 = get_position_id2(sentences_len, bias1)
    test_position_id2 = get_position_id2(test_sentences_len, bias1)

    # 获取所有的词汇
    words = set()
    for each_sentence in sentences:
        words.update(each_sentence.split())
    for each_sentence in test_sentences:
        words.update(each_sentence.split())

    # 组成词典,并获取embedding层的W初始值
    dictionary = dict()
    wordEmbedding = [np.zeros(300, dtype=np.float32)]
    for i, word in enumerate(words):
        dictionary[word] = i + 1
        try:
            wordVec = model.wv[word]
            wordEmbedding.append(wordVec)
        except Exception:
            wordEmbedding.append(-1 + 2 * np.random.random_sample(300))

    # 获取句子向量序列号
    sentences_id = get_sentence_id(sentences, dictionary)
    test_sentences_id = get_sentence_id(test_sentences, dictionary)
    entityPairs_id = get_sentence_id(entityPairs, dictionary)
    test_entityPairs_id = get_sentence_id(test_entityPairs, dictionary)

    position_embedding = np.concatenate((np.random.standard_normal((pos_dict_len, 20)), np.zeros((1, 20))), axis=0)

    with open('./../resource/newDictionary.pickle', 'wb') as f:
        parameter = {
            'dictionary': dictionary,
            'wordEmbedding': wordEmbedding,
            'position_embedding': position_embedding
        }
        train = {
            'sentences_id': sentences_id,
            'sentences_id_len': sentences_id_len,
            'position_id1': position_id1,
            'position_id2': position_id2,
            'entityPairs_id': entityPairs_id,
            'entityPairs_id_len': entityPairs_id_len,
            'labels': labels
        }
        test = {
            'test_sentences_id': test_sentences_id,
            'test_sentences_id_len': test_sentences_id_len,
            'test_position_id1': test_position_id1,
            'test_position_id2': test_position_id2,
            'test_entityPairs_id': test_entityPairs_id,
            'test_entityPairs_id_len': test_entityPairs_id_len,
            'test_labels': test_labels
        }
        pickle.dump(parameter, f, protocol=2)
        pickle.dump(train, f, protocol=2)
        pickle.dump(test, f, protocol=2)

    print("train len ", sentences_max_len)
    print('train pos dic len', pos_dict_len)
    print("train entity len ", entityPairs_max_len)
