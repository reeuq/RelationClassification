import xml.dom.minidom
import string
import numpy as np
from six.moves import cPickle as pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score


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

    with open('./../resource/vec.pickle', 'rb') as f:
        save = pickle.load(f)
        dataset = save['dataset']
        label = save['label']
        del save
        print('dataset', dataset.shape)
        print('label', label.shape)

    c_vec = CountVectorizer(ngram_range=(1, 2))
    X = c_vec.fit_transform(sentences).todense()

    print("X shape", X.shape)

    # X_train = X[:1000, :]
    # y_train = label[:1000]
    #
    # X_test = X[1000:, :]
    # y_test = label[1000:]


    # Instantiate（实例）
    lg = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=42, verbose=1, max_iter=1000, n_jobs=-1)

    scores = cross_val_score(lg, X, label, cv=5, scoring='accuracy')
    print(scores)
    print(np.average(scores))

    # # Fit
    # lg.fit(X_train, y_train)
    #
    # # Predict
    # y_pred = lg.predict(X_test)
    #
    # print(y_test)
    # print(y_pred)
    #
    # from sklearn.metrics import confusion_matrix
    # from sklearn.metrics import classification_report
    # import matplotlib.pyplot as plt
    #
    # classification_report = classification_report(y_test, y_pred)
    # print(classification_report)
    #
    # confusion_matrix = confusion_matrix(y_test, y_pred)
    # print(confusion_matrix)
    #
    # plt.matshow(confusion_matrix)
    # plt.title('混淆矩阵')
    # plt.colorbar()
    # plt.ylabel('实际类型')
    # plt.xlabel('预测类型')
    # plt.show()
    #
    # # Score
    # print(metrics.accuracy_score(y_test, y_pred))