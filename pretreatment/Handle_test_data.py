from gensim.models import word2vec
import xml.dom.minidom
import string
import numpy as np
from six.moves import cPickle as pickle


if __name__ == '__main__':
    model = word2vec.KeyedVectors.load_word2vec_format('./../resource/original/GoogleNews-vectors-negative300.bin', binary=True)

    dom = xml.dom.minidom.parse('./../resource/original/1.1.test.text.xml')
    root = dom.documentElement

    entities = root.getElementsByTagName("entity")
    punc = string.punctuation.replace('-', '').replace('/', '')

    with open('./../resource/original/1.1.test.relations.txt', 'r') as f:
        stringList = f.readlines()
        dataset = np.ndarray(shape=(len(stringList), 300), dtype=np.float32)
        num_raw = 0
        for string_wyd in stringList:
            tab1_string = string_wyd[string_wyd.find('(') + 1:string_wyd.find(',')]
            if string_wyd.find('REVERSE') == -1:
                tab2_string = string_wyd[string_wyd.find(',') + 1:string_wyd.find(')')]
            else:
                tab2_string = string_wyd[string_wyd.find(',') + 1:string_wyd.find(',', string_wyd.find(',') + 1)]
            for entity in entities:
                if entity.getAttribute("id") == tab1_string:
                    tab1_string = entity.firstChild.data
                elif entity.getAttribute("id") == tab2_string:
                    tab2_string = entity.firstChild.data
                    break
            tab1_word_list = tab1_string.translate(str.maketrans('/-', '  ', punc)).split()
            tab1_vec = np.ndarray(shape=(len(tab1_word_list), 300), dtype=np.float32)
            num_vec1 = 0
            for word in tab1_word_list:
                try:
                    wordVec = model.wv[word]
                    tab1_vec[num_vec1, :] = wordVec
                except Exception:
                    tab1_vec[num_vec1, :] = -1 + 2 * np.random.random_sample(300)
                finally:
                    num_vec1 += 1
            tab1 = np.mean(tab1_vec, axis=0)

            tab2_word_list = tab2_string.translate(str.maketrans('/-', '  ', punc)).split()
            tab2_vec = np.ndarray(shape=(len(tab2_word_list), 300), dtype=np.float32)
            num_vec2 = 0
            for word in tab2_word_list:
                try:
                    wordVec = model.wv[word]
                    tab2_vec[num_vec2, :] = wordVec
                except Exception:
                    tab2_vec[num_vec2, :] = -1 + 2 * np.random.random_sample(300)
                finally:
                    num_vec2 += 1
            tab2 = np.mean(tab2_vec, axis=0)
            dataset[num_raw, :] = tab1 - tab2
            num_raw += 1

    with open('./../resource/test_vec.pickle', 'wb') as f:
        save = {
            'test_dataset': dataset,
        }
        pickle.dump(save, f, protocol=2)