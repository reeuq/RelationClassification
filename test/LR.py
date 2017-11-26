from __future__ import print_function

from sklearn.linear_model import LogisticRegression
from six.moves import cPickle as pickle

with open('./../resource/vec.pickle', 'rb') as f:
    save = pickle.load(f)
    dataset = save['dataset']
    label = save['label']
    del save
    print('dataset', dataset.shape)
    print('label', label.shape)

X_train = dataset[:1000, :]
y_train = label[:1000]

X_test = dataset[1000:, :]
y_test = label[1000:]

if __name__=='__main__':
    # Instantiate（实例）
    lg = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=42, verbose=1, max_iter=1000, n_jobs=-1)

    # Fit
    lg.fit(X_train, y_train)

    # Predict
    y_pred = lg.predict(X_test)

    print(y_test)
    print(y_pred)

    # Score
    from sklearn import metrics
    print(metrics.accuracy_score(y_test, y_pred))