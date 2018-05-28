import string
import xml.dom.minidom
from six.moves import cPickle as pickle


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


def get_sentence(sentence, entity, tab2_string):
    if entity.nodeType == 3:
        sentence_demo = sentence + entity.data
        return get_sentence(sentence_demo, entity.nextSibling, tab2_string)
    elif entity.nodeName == "entity":
        if entity.getAttribute("id") == tab2_string:
            # sentence_demo = sentence + entity.firstChild.data
            return sentence
        else:
            sentence_demo = sentence + entity.firstChild.data
            return get_sentence(sentence_demo, entity.nextSibling, tab2_string)
    else:
        return sentence


def get_before_sentence(sentence, entity):
    if entity is not None:
        if entity.nodeType == 3:
            if entity.data.find('.') != -1 or entity.data.find(';') != -1 or \
                            entity.data.find('?') != -1 or entity.data.find('!') != -1:
                sentence_demo = entity.data[entity.data.find('.')+1:] + sentence
                return sentence_demo
            else:
                sentence_demo = entity.data + sentence
                return get_before_sentence(sentence_demo, entity.previousSibling)
        elif entity.nodeName == "entity":
            if entity.firstChild.data.find('.') != -1 or entity.firstChild.data.find(';') != -1 or \
                            entity.firstChild.data.find('?') != -1 or entity.firstChild.data.find('!') != -1:
                sentence_demo = entity.firstChild.data[entity.firstChild.data.find('.')+1:] + sentence
                return sentence_demo
            else:
                sentence_demo = entity.firstChild.data + sentence
                return get_before_sentence(sentence_demo, entity.previousSibling)
        else:
            return sentence
    else:
        return sentence


def get_after_sentence(sentence, entity):
    if entity is not None:
        if entity.nodeType == 3:
            if entity.data.find('.') != -1 or entity.data.find(';') != -1 or \
                            entity.data.find('?') != -1 or entity.data.find('!') != -1:
                sentence_demo = sentence + entity.data[:entity.data.find('.')]
                return sentence_demo
            else:
                sentence_demo = sentence + entity.data
                return get_after_sentence(sentence_demo, entity.nextSibling)
        elif entity.nodeName == "entity":
            if entity.firstChild.data.find('.') != -1 or entity.firstChild.data.find(';') != -1 or \
                            entity.firstChild.data.find('?') != -1 or entity.firstChild.data.find('!') != -1:
                sentence_demo = sentence + entity.firstChild.data[:entity.firstChild.data.find('.')]
                return sentence_demo
            else:
                sentence_demo = sentence + entity.firstChild.data
                return get_after_sentence(sentence_demo, entity.nextSibling)
        else:
            return sentence
    else:
        return sentence


if __name__ == "__main__":
    punc = string.punctuation.replace('-', '').replace('/', '')
    dom = xml.dom.minidom.parse('./../resource/original/1.1.text.xml')
    root = dom.documentElement
    entities = root.getElementsByTagName("entity")

    test_dom = xml.dom.minidom.parse('./../resource/original/1.1.test.text.xml')
    test_root = test_dom.documentElement
    test_entities = test_root.getElementsByTagName("entity")

    sentences = []
    sentences_len = []
    labels = []
    entityPairs = []
    entityPairs_len = []
    with open('./../resource/original/1.1.relations.txt', 'r') as f:
        stringList = f.readlines()
        for string_wyd in stringList:
            labels.append(category(string_wyd))
            tab1_string = string_wyd[string_wyd.find('(') + 1:string_wyd.find(',')]
            if string_wyd.find('REVERSE') == -1:
                tab2_string = string_wyd[string_wyd.find(',') + 1:string_wyd.find(')')]
            else:
                tab2_string = string_wyd[string_wyd.find(',') + 1:string_wyd.find(',', string_wyd.find(',') + 1)]
            for entity in entities:
                if entity.getAttribute("id") == tab1_string:
                    before = get_before_sentence("", entity.previousSibling).translate(
                        str.maketrans('/-', '  ', punc)).strip().lower()
                    entity1 = entity.firstChild.data.translate(str.maketrans('/-', '  ', punc)).strip().lower()
                    mid = get_sentence("", entity.nextSibling, tab2_string).translate(
                        str.maketrans('/-', '  ', punc)).strip().lower()
                if entity.getAttribute("id") == tab2_string:
                    entity2 = entity.firstChild.data.translate(str.maketrans('/-', '  ', punc)).strip().lower()
                    after = get_after_sentence("", entity.nextSibling).translate(
                        str.maketrans('/-', '  ', punc)).strip().lower()
                    sentences_len.append([len(before.split()), len(entity1.split()),
                                          len(mid.split()), len(entity2.split()), len(after.split())])
                    sentences.append(before + ' ' + entity1 + ' ' + mid + ' ' + entity2 + ' ' + after)
                    entityPairs_len.append(len(entity1.split()) + len(entity2.split()))
                    entityPairs.append(entity1 + ' ' + entity2)
                    break

    test_sentences = []
    test_sentences_len = []
    test_labels = []
    test_entityPairs = []
    test_entityPairs_len = []
    with open('./../resource/original/keys.test.1.1.txt', 'r') as f:
        stringList = f.readlines()
        for string_wyd in stringList:
            test_labels.append(category(string_wyd))
            tab1_string = string_wyd[string_wyd.find('(') + 1:string_wyd.find(',')]
            if string_wyd.find('REVERSE') == -1:
                tab2_string = string_wyd[string_wyd.find(',') + 1:string_wyd.find(')')]
            else:
                tab2_string = string_wyd[string_wyd.find(',') + 1:string_wyd.find(',', string_wyd.find(',') + 1)]
            for entity in test_entities:
                if entity.getAttribute("id") == tab1_string:
                    before = get_before_sentence("", entity.previousSibling).translate(
                        str.maketrans('/-', '  ', punc)).strip().lower()
                    entity1 = entity.firstChild.data.translate(str.maketrans('/-', '  ', punc)).strip().lower()
                    mid = get_sentence("", entity.nextSibling, tab2_string).translate(
                        str.maketrans('/-', '  ', punc)).strip().lower()
                if entity.getAttribute("id") == tab2_string:
                    entity2 = entity.firstChild.data.translate(str.maketrans('/-', '  ', punc)).strip().lower()
                    after = get_after_sentence("", entity.nextSibling).translate(
                        str.maketrans('/-', '  ', punc)).strip().lower()
                    test_sentences_len.append([len(before.split()), len(entity1.split()),
                                               len(mid.split()), len(entity2.split()), len(after.split())])
                    test_sentences.append(before + ' ' + entity1 + ' ' + mid + ' ' + entity2 + ' ' + after)
                    test_entityPairs_len.append(len(entity1.split()) + len(entity2.split()))
                    test_entityPairs.append(entity1 + ' ' + entity2)
                    break

    with open('./../resource/newSentence.pickle', 'wb') as f:
        train = {
            'sentences': sentences,
            'sentences_len': sentences_len,
            'labels': labels,
            'entityPairs': entityPairs,
            'entityPairs_len': entityPairs_len
        }
        test = {
            'test_sentences': test_sentences,
            'test_sentences_len': test_sentences_len,
            'test_labels': test_labels,
            'test_entityPairs': test_entityPairs,
            'test_entityPairs_len': test_entityPairs_len
        }
        pickle.dump(train, f, protocol=2)
        pickle.dump(test, f, protocol=2)
    print("end")
