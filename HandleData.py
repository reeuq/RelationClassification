from gensim.models import word2vec
import xml.dom.minidom
import os
import numpy as py

model = word2vec.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
print(model.wv['world'])

print(model.wv.most_similar(positive=['woman', 'king'], negative=['man']))



# dom = xml.dom.minidom.parse('1.1.text.xml')
# root = dom.documentElement
#
# tt = root.getElementsByTagName("doc")
# t = tt[0]
# print(t.childNodes[0].data)
# ss = t.getElementsByTagName("entity")
# s = ss[0]
# print(s.firstChild.data)
# # sentences = word2vec.LineSentence(u"/home/wyd/PycharmProjects/RelationClassification")  # 加载语料
# # model = word2vec.Word2Vec(sentences, size=200)