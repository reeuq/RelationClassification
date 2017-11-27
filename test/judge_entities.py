from gensim.models import word2vec
import xml.dom.minidom
import string

model = word2vec.KeyedVectors.load_word2vec_format('./../resource/GoogleNews-vectors-negative300.bin', binary=True)

dom = xml.dom.minidom.parse('./../resource/1.1.text.xml')
root = dom.documentElement

entities = root.getElementsByTagName("entity")
punc = string.punctuation.replace('-', '').replace('/', '')

errorCount = 0
errormsg = []
for entity in entities:
    entityStr = entity.firstChild.data.translate(str.maketrans('/-', '  ', punc))
    i = 0
    for oneWord in entityStr.split():
        try:
            print(model.wv[oneWord])
            i += 1
        except Exception:
            pass
    if i == 0:
        errorCount += 1
        errormsg.append(entity.getAttribute("id"))

print(errorCount)
print(errormsg)