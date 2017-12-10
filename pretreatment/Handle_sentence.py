from gensim.models import word2vec
import xml.dom.minidom
import string
import numpy as np
from six.moves import cPickle as pickle
from HandleData import category


def get_sentence(sentence, entity, tab2_string):
    if entity.nodeType == 3:
        sentence_demo = sentence + entity.data
        return get_sentence(sentence_demo, entity.nextSibling, tab2_string)
    elif entity.nodeType == 1:
        if entity.getAttribute("id") == tab2_string:
            sentence_demo = sentence + entity.firstChild.data
            return sentence_demo
        else:
            sentence_demo = sentence + entity.firstChild.data
            return get_sentence(sentence_demo, entity.nextSibling, tab2_string)


if __name__ == "__main__":
    model = word2vec.KeyedVectors.load_word2vec_format('./../resource/GoogleNews-vectors-negative300.bin', binary=True)

    dom = xml.dom.minidom.parse('./../resource/1.1.text.xml')
    root = dom.documentElement

    entities = root.getElementsByTagName("entity")
    punc = string.punctuation.replace('-', '').replace('/', '')

    sentences = []
    with open('./../resource/1.1.relations.txt', 'r') as f:
        stringList = f.readlines()
        for string_wyd in stringList:
            tab1_string = string_wyd[string_wyd.find('(') + 1:string_wyd.find(',')]
            if string_wyd.find('REVERSE') == -1:
                tab2_string = string_wyd[string_wyd.find(',') + 1:string_wyd.find(')')]
            else:
                tab2_string = string_wyd[string_wyd.find(',') + 1:string_wyd.find(',', string_wyd.find(',') + 1)]
            for entity in entities:
                if entity.getAttribute("id") == tab1_string:
                    sentences.append(get_sentence("", entity, tab2_string))
                    break

    dataset = np.zeros(shape=(len(sentences), 35, 300), dtype=np.float32)
    num_sentence = 0
    for sentence in sentences:
        word_list = sentence.translate(str.maketrans('/-', '  ', punc)).split()
        num_word = 0
        for word in word_list:
            try:
                wordVec = model.wv[word]
                dataset[num_sentence, num_word, :] = wordVec
            except Exception:
                dataset[num_sentence, num_word, :] = -1 + 2 * np.random.random_sample(300)
            finally:
                num_word += 1
        num_sentence += 1

    with open('./../resource/sentenceVec.pickle', 'wb') as f:
        save = {
            'dataset': dataset
        }
        pickle.dump(save, f, protocol=2)
    print(dataset.shape)