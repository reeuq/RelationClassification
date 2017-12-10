from __future__ import print_function

import numpy as np
import tensorflow as tf
from keras.preprocessing import sequence

from six.moves import cPickle as pickle
# from evaluation.Evaluation import accuracy
# from evaluation.Evaluation import precision_each_class
# from evaluation.Evaluation import recall_each_class
# from evaluation.Evaluation import f1_each_class_precision_recall


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])

pickle_file = './../resource/vec.pickle'
with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    dataset = save['dataset']
    label = save['label']
    del save
    print('dataset', dataset.shape)
    print('label', label.shape)

pickle_file = './../resource/sentenceVec.pickle'
with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    sentenceDataset = save['dataset']
    del save
    print('sentenceDataset', sentenceDataset.shape)


def reformat(labels):
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(11) == labels[:, None]).astype(np.float32)
    return labels
label = reformat(label)
print('label', label.shape)


train_dataset = dataset[:700, :]
train_sentence_dataset = sentenceDataset[:700, :, :]
train_labels = label[:700, :]
valid_dataset = dataset[700:1000, :]
valid_sentence_dataset = sentenceDataset[700:1000, :, :]
valid_labels = label[700:1000, :]
test_dataset = dataset[1000:, :]
test_sentence_dataset = sentenceDataset[1000:, :, :]
test_labels = label[1000:, :]
print('Training set', train_dataset.shape, train_sentence_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_sentence_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_sentence_dataset.shape, test_labels.shape)

sen1_train = sequence.pad_sequences(sen1_train,maxlen=max_len,truncating='post',padding='post')


graph = tf.Graph()
with graph.as_default():
    tf_train = tf.constant(train_dataset)
    tf_train_sentence_dataset = tf.constant(train_sentence_dataset)
    tf_valid = tf.constant(valid_dataset)
    tf_valid_sentence_dataset = tf.constant(valid_sentence_dataset)
    tf_test = tf.constant(test_dataset)
    tf_test_sentence_dataset = tf.constant(test_sentence_dataset)

    tf_train_labels = tf.constant(train_labels)

    sentence_node = 300
    W1 = tf.Variable(tf.truncated_normal([300, sentence_node]))

    X1_train_dataset = tf.reshape(tf_train_sentence_dataset, [-1, 300])
    X1_valid_dataset = tf.reshape(tf_valid_sentence_dataset, [-1, 300])
    X1_test_dataset = tf.reshape(tf_test_sentence_dataset, [-1, 300])

    train_Z_dataset = tf.matmul(X1_train_dataset, W1)
    train_Z_dataset_reshape = tf.reshape(train_Z_dataset, [-1, 35, sentence_node])
    train_Z_dataset_transpose = tf.transpose(train_Z_dataset_reshape, perm=[0, 1, 2])
    train_m_dataset = tf.reduce_max(train_Z_dataset_transpose, 2)

    valid_Z_dataset = tf.matmul(X1_valid_dataset, W1)
    valid_Z_dataset_reshape = tf.reshape(valid_Z_dataset, [-1, 35, sentence_node])
    valid_Z_dataset_transpose = tf.transpose(valid_Z_dataset_reshape, perm=[0, 1, 2])
    valid_m_dataset = tf.reduce_max(valid_Z_dataset_transpose, 2)

    test_Z_dataset = tf.matmul(X1_test_dataset, W1)
    test_Z_dataset_reshape = tf.reshape(test_Z_dataset, [-1, 35, sentence_node])
    test_Z_dataset_transpose = tf.transpose(test_Z_dataset_reshape, perm=[0, 1, 2])
    test_m_dataset = tf.reduce_max(test_Z_dataset_transpose, 2)

    # trans_node_count = 300
    # W2 = tf.Variable(tf.truncated_normal([sentence_node, trans_node_count]))
    # B2 = tf.Variable(tf.zeros([trans_node_count]))
    #
    # train_Z = tf.matmul(train_m_dataset, W2) + B2
    # train_m = tf.tanh(train_Z)
    #
    # valid_Z = tf.matmul(valid_m_dataset, W2) + B2
    # valid_m = tf.tanh(valid_Z)
    #
    # test_Z = tf.matmul(test_m_dataset, W2) + B2
    # test_m = tf.tanh(test_Z)

    # tf_train_dataset = tf.concat([train_m, tf_train], 1)
    # tf_valid_dataset = tf.concat([valid_m, tf_valid], 1)
    # tf_test_dataset = tf.concat([test_m, tf_test], 1)

    tf_train_dataset = train_m_dataset
    tf_valid_dataset = valid_m_dataset
    tf_test_dataset = test_m_dataset

    hidden_node_count = 1024
    # Variables.
    weights1 = tf.Variable(tf.truncated_normal([35, hidden_node_count]))
    biases1 = tf.Variable(tf.zeros([hidden_node_count]))

    weights2 = tf.Variable(tf.truncated_normal([hidden_node_count, 11]))
    biases2 = tf.Variable(tf.zeros([11]))

    # Training computation. right most
    ys = tf.matmul(tf_train_dataset, weights1) + biases1
    hidden = tf.nn.relu(ys)
    h_fc = hidden

    valid_y0 = tf.matmul(tf_valid_dataset, weights1) + biases1
    valid_hidden1 = tf.nn.relu(valid_y0)

    test_y0 = tf.matmul(tf_test_dataset, weights1) + biases1
    test_hidden1 = tf.nn.relu(test_y0)

    # enable DropOut
    keep_prob = tf.placeholder(tf.float32)
    hidden_drop = tf.nn.dropout(hidden, keep_prob)
    h_fc = hidden_drop

    # left most
    logits = tf.matmul(h_fc, weights2) + biases2
    # only drop out when train
    logits_predict = tf.matmul(hidden, weights2) + biases2
    valid_predict = tf.matmul(valid_hidden1, weights2) + biases2
    test_predict = tf.matmul(test_hidden1, weights2) + biases2
    # loss
    l2_loss = tf.nn.l2_loss(weights1) + tf.nn.l2_loss(biases1) + tf.nn.l2_loss(weights2) + tf.nn.l2_loss(biases2)
    # enable regularization
    beta = 0.002
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels)) + beta * l2_loss

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits_predict)
    valid_prediction = tf.nn.softmax(valid_predict)
    test_prediction = tf.nn.softmax(test_predict)

num_steps = 2001

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Initialized")
    for step in range(num_steps):
        # Run the computations. We tell .run() that we want to run the optimizer,
        # and get the loss value and the training predictions returned as numpy
        # arrays.
        feed_dict = {keep_prob: 0.5}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 100 == 0):
            print('Loss at step %d: %f' % (step, l))
            print('Training accuracy: %.1f%%' % accuracy(predictions, train_labels))
            print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))
    print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))

    # print('train_Z', train_Z.shape)
    # print("train_m", train_m.shape)
    #
    # print('---------------------------------------------')
    # precision = precision_each_class(test_prediction.eval(), test_labels)
    # print(precision)
    # print('---------------------------------------------')
    # recall = recall_each_class(test_prediction.eval(), test_labels)
    # print(recall)
    # print('---------------------------------------------')
    # print(f1_each_class_precision_recall(precision, recall))