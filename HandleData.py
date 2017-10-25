from gensim.models import word2vec
import xml.dom.minidom
import os
import string
import numpy as py

model = word2vec.KeyedVectors.load_word2vec_format('./resource/GoogleNews-vectors-negative300.bin', binary=True)

dom = xml.dom.minidom.parse('./resource/1.1.text.xml')
root = dom.documentElement

entities = root.getElementsByTagName("entity")
allCount = 0
errorMsg = []
errorCount = 0
allSets = set()
words = []
for entity in entities:
    punc = string.punctuation.replace('-','').replace('/','')
    entity = entity.firstChild.data.translate(str.maketrans('/-', '  ', punc))
    words.extend(entity.split())
    # allSets.update(words)
for oneWord in words:
    allCount+=1
    print(oneWord)
    try:
        print(model.wv[oneWord])
    except Exception:
        errorCount+=1
        errorMsg.append(oneWord)

print(allCount)
print(errorCount)
print(errorMsg)