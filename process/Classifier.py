from __future__ import print_function

import numpy as np
import tensorflow as tf
from keras.preprocessing import sequence
import sys,os
from six.moves import cPickle as pickle

sys.path.append(os.path.join(os.path.dirname(__file__),'..'))

from evaluation.Evaluation import accuracy
from evaluation.Evaluation import precision_each_class
from evaluation.Evaluation import recall_each_class
from evaluation.Evaluation import f1_each_class_precision_recall
from evaluation.Evaluation import class_label_count


def print_out(precision, recall, f1, count):
    print("%24s%10s%10s%10s%10s"%("","precision","recall","f1","count"))
    print("%24s%10.2f%10.2f%10.2f%10d" % ("usage", precision[0], recall[0], f1[0], count[0]))
    print("%24s%10.2f%10.2f%10.2f%10d" % ("usage_reverse", precision[1], recall[1], f1[1], count[1]))
    print("%24s%10.2f%10.2f%10.2f%10d" % ("result", precision[2], recall[2], f1[2], count[2]))
    print("%24s%10.2f%10.2f%10.2f%10d" % ("result_reverse", precision[3], recall[3], f1[3], count[3]))
    print("%24s%10.2f%10.2f%10.2f%10d" % ("model_feature", precision[4], recall[4], f1[4], count[4]))
    print("%24s%10.2f%10.2f%10.2f%10d" % ("model_feature_reverse", precision[5], recall[5], f1[5], count[5]))
    print("%24s%10.2f%10.2f%10.2f%10d" % ("part_whole", precision[6], recall[6], f1[6], count[6]))
    print("%24s%10.2f%10.2f%10.2f%10d" % ("part_whole_reverse", precision[7], recall[7], f1[7], count[7]))
    print("%24s%10.2f%10.2f%10.2f%10d" % ("topic", precision[8], recall[8], f1[8], count[8]))
    print("%24s%10.2f%10.2f%10.2f%10d" % ("topic_reverse", precision[9], recall[9], f1[9], count[9]))
    print("%24s%10.2f%10.2f%10.2f%10d" % ("compare", precision[10], recall[10], f1[10], count[10]))
    print("%24s%10.2f%10.2f%10.2f%10d" % ("average/sum", np.mean(precision), np.mean(recall), mean(f1), sum(count)))

pickle_file = './../resource/vec.pickle'
with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    dataset = save['dataset']
    label = save['label']
    del save
    print('dataset', dataset.shape)
    print('label', label.shape)

# pickle_file = './../resource/sentenceVec.pickle'
# with open(pickle_file, 'rb') as f:
#     save = pickle.load(f)
#     sentenceDataset = save['dataset']
#     del save
#     print('sentenceDataset', sentenceDataset.shape)


pickle_file = './../resource/dictionary.pickle'
with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    dictionary = save['dictionary']
    W = save['W']
    sentences_vec = save['sentences_vec']
    del save


def reformat(labels):
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(11) == labels[:, None]).astype(np.float32)
    return labels
label = reformat(label)
print('label', label.shape)


def randomize(dataset, sentence_dataset, labels, sentence_dataset_len):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :]
    shuffled_labels = labels[permutation, :]
    shuffled_sentence_dataset = sentence_dataset[permutation, :]
    shuffled_sentence_dataset_len = sentence_dataset_len[permutation]
    return shuffled_dataset, shuffled_sentence_dataset, shuffled_labels, shuffled_sentence_dataset_len

W = np.array(W)
sen1_train = sequence.pad_sequences(sentences_vec, maxlen=29, truncating='post', padding='post')
sen1_train_len = np.array([len(s) for s in sentences_vec])

dataset, sen1_train, label, sen1_train_len = randomize(dataset, sen1_train, label, sen1_train_len)

train_dataset = dataset[:900, :]
train_sentence_dataset = sen1_train[:900, :]
train_sentence_dataset_len = sen1_train_len[:900]
train_labels = label[:900, :]

valid_dataset = dataset[900:1100, :]
valid_sentence_dataset = sen1_train[900:1100, :]
valid_sentence_dataset_len = sen1_train_len[900:1100]
valid_labels = label[900:1100, :]

test_dataset = dataset[1100:, :]
test_sentence_dataset = sen1_train[1100:, :]
test_sentence_dataset_len = sen1_train_len[1100:]
test_labels = label[1100:, :]

print('Training set', train_dataset.shape, train_sentence_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_sentence_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_sentence_dataset.shape, test_labels.shape)


batch_size = 96

graph = tf.Graph()
with graph.as_default():
    # tf_train = tf.constant(train_dataset)
    # tf_train_sentence_dataset = tf.constant(train_sentence_dataset, dtype=tf.int32)
    # tf_train_sentence_dataset_len = tf.constant(train_sentence_dataset_len, dtype=tf.int32)
    tf_train = tf.placeholder(tf.float32, shape=(batch_size, 300))
    tf_train_sentence_dataset = tf.placeholder(tf.int32, shape=(batch_size, 29))
    tf_train_sentence_dataset_len = tf.placeholder(tf.int32, shape=(batch_size))

    tf_valid = tf.constant(valid_dataset)
    tf_valid_sentence_dataset = tf.constant(valid_sentence_dataset, dtype=tf.int32)
    tf_valid_sentence_dataset_len = tf.constant(valid_sentence_dataset_len, dtype=tf.int32)

    tf_test = tf.constant(test_dataset)
    tf_test_sentence_dataset = tf.constant(test_sentence_dataset, dtype=tf.int32)
    tf_test_sentence_dataset_len = tf.constant(test_sentence_dataset_len, dtype=tf.int32)

    # tf_train_labels = tf.constant(train_labels)
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, 11))

    W_vec = tf.Variable(W, dtype=tf.float32, trainable=True)
    train_embedded = tf.nn.embedding_lookup(W_vec, tf_train_sentence_dataset)
    valid_embedded = tf.nn.embedding_lookup(W_vec, tf_valid_sentence_dataset)
    test_embedded = tf.nn.embedding_lookup(W_vec, tf_test_sentence_dataset)

    # sentence_node = 300
    # W1 = tf.Variable(tf.truncated_normal([300, sentence_node]))
    #
    # X1_train_dataset = tf.reshape(train_embedded, [-1, 300])
    # X1_valid_dataset = tf.reshape(valid_embedded, [-1, 300])
    # X1_test_dataset = tf.reshape(test_embedded, [-1, 300])
    #
    # train_Z_dataset = tf.matmul(X1_train_dataset, W1)
    # train_Z_dataset_reshape = tf.reshape(train_Z_dataset, [-1, 35, sentence_node])
    # train_Z_dataset_transpose = tf.transpose(train_embedded, perm=[0, 2, 1])
    # train_m_dataset = tf.reduce_max(train_Z_dataset_transpose, 2)

    # valid_Z_dataset = tf.matmul(X1_valid_dataset, W1)
    # valid_Z_dataset_reshape = tf.reshape(valid_Z_dataset, [-1, 35, sentence_node])
    # valid_Z_dataset_transpose = tf.transpose(valid_embedded, perm=[0, 2, 1])
    # valid_m_dataset = tf.reduce_max(valid_Z_dataset_transpose, 2)

    # test_Z_dataset = tf.matmul(X1_test_dataset, W1)
    # test_Z_dataset_reshape = tf.reshape(test_Z_dataset, [-1, 35, sentence_node])
    # test_Z_dataset_transpose = tf.transpose(test_embedded, perm=[0, 2, 1])
    # test_m_dataset = tf.reduce_max(test_Z_dataset_transpose, 2)

    def get_last(inputs, seq_len):
        batch_size = tf.shape(inputs)[0]
        max_seq_len = tf.shape(inputs)[1]
        dim = inputs.get_shape().as_list()[2]
        index_list = tf.range(batch_size) * max_seq_len + (seq_len - 1)
        last_outputs = tf.gather(tf.reshape(inputs, (-1, dim)), index_list, name='last_outputs')
        return last_outputs

    sen1_cell_train = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(300, state_is_tuple=True),
                                              input_keep_prob=0.5, output_keep_prob=0.5)
    sen1_outputs_train, sen1_states_train = tf.nn.dynamic_rnn(sen1_cell_train, train_embedded, sequence_length=tf_train_sentence_dataset_len, dtype=tf.float32)
    # sen1_last_output_train = get_last(sen1_outputs_train, tf_train_sentence_dataset_len)
    sen1_last_output_train = sen1_states_train.h

    sen1_cell_valid = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(300, state_is_tuple=True, reuse=True),
                                              input_keep_prob=1, output_keep_prob=1)
    sen1_outputs_valid, sen1_states_valid = tf.nn.dynamic_rnn(sen1_cell_valid, valid_embedded, sequence_length=tf_valid_sentence_dataset_len, dtype=tf.float32)
    # sen1_last_output_valid = get_last(sen1_outputs_valid,tf_valid_sentence_dataset_len)
    sen1_last_output_valid = sen1_states_valid.h

    sen1_cell_test = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(300, state_is_tuple=True, reuse=True),
                                              input_keep_prob=1, output_keep_prob=1)
    sen1_outputs_test, sen1_states_test = tf.nn.dynamic_rnn(sen1_cell_test, test_embedded, sequence_length=tf_test_sentence_dataset_len, dtype=tf.float32)
    # sen1_last_output_test = get_last(sen1_outputs_test, tf_test_sentence_dataset_len)
    sen1_last_output_test = sen1_states_test.h

    tf_train_dataset = tf.concat([sen1_last_output_train, tf_train], 1)
    tf_valid_dataset = tf.concat([sen1_last_output_valid, tf_valid], 1)
    tf_test_dataset = tf.concat([sen1_last_output_test, tf_test], 1)

    # tf_train_dataset = sen1_last_output_train
    # tf_valid_dataset = sen1_last_output_valid
    # tf_test_dataset = sen1_last_output_test
    #
    # tf_train_dataset = tf_train
    # tf_valid_dataset = tf_valid
    # tf_test_dataset = tf_test

    hidden_node_count = 300
    # Variables.
    weights1 = tf.Variable(tf.truncated_normal([600, hidden_node_count]))
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
    optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits_predict)
    valid_prediction = tf.nn.softmax(valid_predict)
    test_prediction = tf.nn.softmax(test_predict)

num_steps = 3001

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Initialized")
    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_sentence_data = train_sentence_dataset[offset:(offset + batch_size), :]
        batch_sentence_data_len = train_sentence_dataset_len[offset:(offset + batch_size)]

        batch_labels = train_labels[offset:(offset + batch_size), :]

        feed_dict = {tf_train: batch_data, tf_train_sentence_dataset: batch_sentence_data,
                     tf_train_sentence_dataset_len: batch_sentence_data_len,
                     tf_train_labels: batch_labels, keep_prob: 0.5}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 100 == 0):
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
            print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))
    print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))

    print('---------------------------------------------')
    precision = precision_each_class(test_prediction.eval(), test_labels)
    recall = recall_each_class(test_prediction.eval(), test_labels)
    f1 = f1_each_class_precision_recall(precision, recall)
    count = class_label_count(test_labels)
    print_out(precision, recall, f1, count)
