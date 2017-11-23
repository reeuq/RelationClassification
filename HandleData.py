from gensim.models import word2vec
import xml.dom.minidom
import os
import string
import numpy as np
from six.moves import cPickle as pickle

model = word2vec.KeyedVectors.load_word2vec_format('./resource/GoogleNews-vectors-negative300.bin', binary=True)

dom = xml.dom.minidom.parse('./resource/1.1.text.xml')
root = dom.documentElement

entities = root.getElementsByTagName("entity")
punc = string.punctuation.replace('-', '').replace('/', '')

# allCount = 0
# errorMsg = []
# errorCount = 0
# allSets = set()
# words = []
# for entity in entities:
#     entity = entity.firstChild.data.translate(str.maketrans('/-', '  ', punc))
#     words.extend(entity.split())
#     # allSets.update(words)
# for oneWord in words:
#     allCount+=1
#     print(oneWord)
#     try:
#         print(model.wv[oneWord])
#     except Exception:
#         errorCount+=1
#         errorMsg.append(oneWord)
#
# print(allCount)
# print(errorCount)
# print(errorMsg)


def category(str1):
    if str1[:str1.find('(')] == "USAGE":
        if str1.find('REVERSE') == -1:
            return 0
        else:
            return 1
    elif str1[:str1.find('(')] == "RESULT":
        if str1.find('REVERSE') == -1:
            return 2
        else:
            return 3
    elif str1[:str1.find('(')] == "MODEL-FEATURE":
        if str1.find('REVERSE') == -1:
            return 4
        else:
            return 5
    elif str1[:str1.find('(')] == "PART_WHOLE":
        if str1.find('REVERSE') == -1:
            return 6
        else:
            return 7
    elif str1[:str1.find('(')] == "TOPIC":
        if str1.find('REVERSE') == -1:
            return 8
        else:
            return 9
    elif str1[:str1.find('(')] == "COMPARE":
        return 10


with open('./resource/1.1.relations.txt','r') as f:
    stringList = f.readlines()
    dataset = np.ndarray(shape=(len(stringList), 300), dtype=np.float32)
    label = np.ndarray(shape=(len(stringList)))
    num_raw = 0
    for string_wyd in stringList:
        label[num_raw] = category(string_wyd)

        tab1_string = string_wyd[string_wyd.find('(')+1:string_wyd.find(',')]
        if string_wyd.find('REVERSE') == -1:
            tab2_string = string_wyd[string_wyd.find(',')+1:string_wyd.find(')')]
        else:
            tab2_string = string_wyd[string_wyd.find(',')+1:string_wyd.find(',',string_wyd.find(',')+1)]
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
                tab1_vec[num_vec1, :] = np.zeros(300)
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

with open('./resource/vec.pickle', 'wb') as f:
    save = {
        'dataset': dataset,
        'label': label
    }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)