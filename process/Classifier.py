# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import tensorflow as tf
from keras.preprocessing import sequence
import sys
import os
from six.moves import cPickle as pickle

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from evaluation.Evaluation import accuracy
from evaluation.Evaluation import precision_each_class
from evaluation.Evaluation import recall_each_class
from evaluation.Evaluation import f1_each_class_precision_recall
from evaluation.Evaluation import class_label_count
from evaluation.Evaluation import print_out


def reformat(labels):
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(11) == labels[:, None]).astype(np.float32)
    return labels


class Model(object):
    def __init__(self, sen_max_len, words_vec, lstm_num_units, n_class, entity_vec_dim, hidden_dim):
        # placeholder
        self.entity_vec = tf.placeholder(tf.float32, [None, entity_vec_dim], name='entity_vec')
        self.sen_ids = tf.placeholder(tf.int32, [None, sen_max_len], name='sen_ids')
        self.sen_len = tf.placeholder(tf.int32, [None], name='sen_len')

        self.labels = tf.placeholder(tf.float32, [None, n_class], name='labels')
        self.lstm_input_keep_prob = tf.placeholder(tf.float32, name='lstm_input_keep_prob')
        self.lstm_output_keep_prob = tf.placeholder(tf.float32, name='lstm_output_keep_prob')
        self.hidden_keep_prob = tf.placeholder(tf.float32, name='hidden_keep_prob')
        self.l2_loss_beta = tf.placeholder(tf.float32, name='l2_loss_beta')
        # Embedding Layer
        # get embedding
        # with tf.device('/cpu:0'), tf.name_scope('embedding'):
        W_dic = tf.Variable(words_vec, dtype=tf.float32, trainable=True, name='W_dic')
        self.sen_embedded = tf.nn.embedding_lookup(W_dic, self.sen_ids, name='sen_embedded')

        def get_last(inputs, seq_len):
            batch_size = tf.shape(inputs)[0]
            max_seq_len = tf.shape(inputs)[1]
            dim = inputs.get_shape().as_list()[2]
            index_list = tf.range(batch_size) * max_seq_len + (seq_len - 1)
            last_outputs = tf.gather(tf.reshape(inputs, (-1, dim)), index_list, name='last_outputs')
            return last_outputs

        # LSTM layer
        with tf.variable_scope('sen'):
            sen_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(lstm_num_units, state_is_tuple=True),
                                                     input_keep_prob=self.lstm_input_keep_prob,
                                                     output_keep_prob=self.lstm_output_keep_prob)
            sen_outputs, sen_states = tf.nn.dynamic_rnn(sen_cell, self.sen_embedded,
                                                        sequence_length=self.sen_len, dtype=tf.float32)
            sen_last_output = sen_states.h

        self.merge = tf.concat([sen_last_output, self.entity_vec], 1)

        W1 = tf.Variable(tf.truncated_normal([lstm_num_units + entity_vec_dim, hidden_dim]))
        b1 = tf.Variable(tf.zeros([hidden_dim]))
        output = tf.nn.relu(tf.nn.xw_plus_b(self.merge, W1, b1))

        W2 = tf.Variable(tf.truncated_normal([hidden_dim, n_class]))
        b2 = tf.Variable(tf.zeros([n_class]))
        logits = tf.nn.xw_plus_b(tf.nn.dropout(output, self.hidden_keep_prob), W2, b2)
        l2_loss = tf.nn.l2_loss(W1) + tf.nn.l2_loss(b1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(b2)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.labels)) \
                    + self.l2_loss_beta * l2_loss
        self.prob = tf.nn.softmax(tf.nn.xw_plus_b(output, W2, b2))


if __name__ == '__main__':
    # --------------------model-------------------------
    # lstm 长度
    sen_max_len = 29
    # lstm 单元维度（细胞输出维度）
    lstm_num_units = 300
    # 实体字典的维度
    entity_vec_dim = 300
    # 隐藏层节点数
    hidden_dim = 300
    # 类别个数
    n_class = 11
    # dropout 概率
    lstm_input_keep_prob = 0.5
    lstm_output_keep_prob = 0.5
    hidden_keep_prob = 0.5
    # train
    # 一批传入多少数据
    batch_size = 96
    # 训练总步数
    num_steps = 5001
    # 每训练多少批次，使用develop数据集验证
    evaluate_every = 100
    # 存储模型个数
    num_checkpoints = 5
    # 初始学习率
    starter_learning_rate = 0.2
    # l2正则项系数
    l2_loss_beta = 0.002
    # 是否为训练阶段
    is_train = False

    print('load data........')
    with open('./../resource/vec.pickle', 'rb') as f:
        save = pickle.load(f)
        dataset = save['dataset']
        label = save['label']
        del save
    with open('./../resource/test_vec.pickle', 'rb') as f:
        save = pickle.load(f)
        test_dataset = save['test_dataset']
        del save
    with open('./../resource/dictionary.pickle', 'rb') as f:
        save = pickle.load(f)
        W = save['W']
        sentences_vec = save['sentences_vec']
        test_sentences_vec = save['test_sentences_vec']
        del save
    # 将label转为one-hot编码形式
    label = reformat(label)
    # 将vector由list形式转换为ndarray形式
    vector = np.array(W)
    sentences_ids = sequence.pad_sequences(sentences_vec, maxlen=sen_max_len, truncating='post', padding='post')
    test_sentences_ids = sequence.pad_sequences(test_sentences_vec, maxlen=sen_max_len, truncating='post', padding='post')
    sentences_ids_len = np.array([len(s) for s in sentences_vec])
    test_sentences_ids_len = np.array([len(s) for s in test_sentences_vec])
    # 分割训练集和验证集
    train_word_vec = dataset[157:, :]
    train_sentences_ids = sentences_ids[157:, :]
    train_sentences_ids_len = sentences_ids_len[157:]
    train_labels = label[157:, :]

    valid_word_vec = dataset[:157, :]
    valid_sentences_ids = sentences_ids[:157, :]
    valid_sentences_ids_len = sentences_ids_len[:157]
    valid_labels = label[:157, :]

    print('Training set', train_word_vec.shape, train_sentences_ids.shape, train_sentences_ids_len.shape,
          train_labels.shape)
    print('Validation set', valid_word_vec.shape, valid_sentences_ids.shape, valid_sentences_ids_len.shape,
          valid_labels.shape)
    print('Test set', test_dataset.shape, test_sentences_ids.shape, test_sentences_ids_len.shape)

    with tf.Session() as sess:
        model = Model(sen_max_len=sen_max_len, words_vec=vector, lstm_num_units=lstm_num_units,
                      n_class=n_class, entity_vec_dim=entity_vec_dim, hidden_dim=hidden_dim)
        # 记录全局步数
        global_step = tf.Variable(0, trainable=False, name='global_step')
        # 选择使用的优化器
        # learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 500, 1, staircase=True)
        train_op = tf.train.GradientDescentOptimizer(starter_learning_rate).minimize(model.loss, global_step=global_step)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=num_checkpoints)
        sess.run(tf.global_variables_initializer())

        if is_train:
            max_acc = 0
            print('Start training.....')
            for step in range(num_steps):
                offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
                batch_word_vec = train_word_vec[offset:(offset + batch_size), :]
                batch_sentence_ids = train_sentences_ids[offset:(offset + batch_size), :]
                batch_sentence_ids_len = train_sentences_ids_len[offset:(offset + batch_size)]

                batch_labels = train_labels[offset:(offset + batch_size), :]

                feed_dict = {model.entity_vec: batch_word_vec,
                             model.sen_ids: batch_sentence_ids,
                             model.sen_len: batch_sentence_ids_len,
                             model.labels: batch_labels,
                             model.lstm_input_keep_prob: lstm_input_keep_prob,
                             model.lstm_output_keep_prob: lstm_output_keep_prob,
                             model.hidden_keep_prob: hidden_keep_prob,
                             model.l2_loss_beta: l2_loss_beta}
                _, loss, predictions = sess.run([train_op, model.loss, model.prob], feed_dict=feed_dict)
                if step % evaluate_every == 0:
                    print("Minibatch loss at step %d: %f" % (step, loss))
                    print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
                    feed_dic = {model.entity_vec: valid_word_vec,
                                model.sen_ids: valid_sentences_ids,
                                model.sen_len: valid_sentences_ids_len,
                                model.labels: valid_labels,
                                model.lstm_input_keep_prob: 1,
                                model.lstm_output_keep_prob: 1,
                                model.hidden_keep_prob: 1,
                                model.l2_loss_beta: l2_loss_beta}
                    valid_predictions = sess.run(model.prob, feed_dict=feed_dic)
                    predict_accuracy = accuracy(valid_predictions, valid_labels)
                    if predict_accuracy > max_acc:
                        max_acc = predict_accuracy
                        saver.save(sess, './../resource/model/classifier.ckpt')
                    print("Validation accuracy: %.1f%%" % predict_accuracy)
            print('---------------------------------------------')
            precision = precision_each_class(valid_predictions, valid_labels)
            recall = recall_each_class(valid_predictions, valid_labels)
            f1 = f1_each_class_precision_recall(precision, recall)
            count = class_label_count(valid_labels)
            print_out(precision, recall, f1, count)
        else:
            model_file = tf.train.latest_checkpoint('./../resource/model/')
            saver.restore(sess, model_file)
            feed_dic = {model.entity_vec: test_dataset,
                        model.sen_ids: test_sentences_ids,
                        model.sen_len: test_sentences_ids_len,
                        model.lstm_input_keep_prob: 1,
                        model.lstm_output_keep_prob: 1,
                        model.hidden_keep_prob: 1,
                        model.l2_loss_beta: l2_loss_beta}
            test_predictions = sess.run(model.prob, feed_dict=feed_dic)
            test_result = np.argmax(test_predictions, 1)
            label = []
            for result in test_result:
                if result == 0 or result == 1:
                    label.append("USAGE")
                elif result ==2 or result == 3:
                    label.append("RESULT")
                elif result == 4 or result == 5:
                    label.append("MODEL-FEATURE")
                elif result == 6 or result == 7:
                    label.append("PART_WHOLE")
                elif result == 8 or result == 9:
                    label.append("TOPIC")
                elif result == 10:
                    label.append("COMPARE")
            final_result = []
            with open('./../resource/1.1.test.relations.txt', 'r') as f:
                stringList = f.readlines()
                for i, string_wyd in enumerate(stringList):
                    final_result.append(label[i]+string_wyd)
            with open('./../resource/result.txt', 'w') as f:
                for result in final_result:
                    f.write(result)