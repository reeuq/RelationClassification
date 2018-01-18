import string
import xml.dom.minidom
from six.moves import cPickle as pickle


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
    dom = xml.dom.minidom.parse('./../resource/original/1.1.text.xml')
    root = dom.documentElement

    entities = root.getElementsByTagName("entity")
    punc = string.punctuation.replace('-', '').replace('/', '')

    sentences = []
    with open('./../resource/original/1.1.relations.txt', 'r') as f:
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
    with open('./../resource/sentence.pickle', 'wb') as f:
        save = {
            'sentences': sentences
        }
        pickle.dump(save, f, protocol=2)
    print("end")