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

# 相对位置特征词典大小
pos_dic_max_len = 121
# 相对位置特征维度
pos_vec_dim = 20
# 句子最大长度
sen_max_len = 73
# 实体对最大长度
entity_max_len = 17
# gru单元维度（细胞输出维度）
gru_num_units = 300
# 隐藏层单元维度
hidden_dim = 300
# 类别个数
n_class = 11
# dropout 概率
gru_input_keep_prob = 0.5
gru_output_keep_prob = 0.5
hidden_keep_prob = 0.5
# l2正则项系数
l2_loss_beta = 0.001
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
starter_learning_rate = 0.1
# 是否为训练阶段
is_train = True


def reformat(labels):
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(11) == labels[:, None]).astype(np.float32)
    return labels


class Model(object):
    def __init__(self, is_training, embedding_vector):
        # placeholder
        # self.entity_ids = tf.placeholder(tf.int32, [None, entity_max_len], name='entity_ids')
        # self.entity_ids_len = tf.placeholder(tf.int32, [None], name='entity_len')
        # self.sen_ids = tf.placeholder(tf.int32, [None, sen_max_len], name='sen_ids')
        # self.pos1_ids = tf.placeholder(tf.int32, [None, sen_max_len], name='pos1_ids')
        # self.pos2_ids = tf.placeholder(tf.int32, [None, sen_max_len], name='pos2_ids')
        # self.sen_len = tf.placeholder(tf.int32, [None], name='sen_len')

        self.entity_vec = tf.placeholder(tf.float32, [None, 300], name='entity_vec')

        self.labels = tf.placeholder(tf.int32, [None, n_class], name='labels')
        # Embedding Layer
        # get embedding
        # with tf.device('/cpu:0'), tf.name_scope('embedding'):
        #     wordEmbedding = tf.Variable(embedding_vector, dtype=tf.float32, trainable=False, name='wordEmbedding')
            # self.sen_embedded = tf.nn.embedding_lookup(wordEmbedding, self.sen_ids, name='sen_embedded')
            # self.entity_embedded = tf.nn.embedding_lookup(wordEmbedding, self.entity_ids, name='entity_embedded')
            # vector = tf.concat([tf.truncated_normal([pos_dic_max_len, pos_vec_dim]), tf.zeros([1, pos_vec_dim])], 0)
            # self.posEmbedding = tf.Variable(vector, dtype=tf.float32, trainable=True, name='posEmbedding')
            # self.pos1_embedded = tf.nn.embedding_lookup(self.posEmbedding, self.pos1_ids, name='pos1_embedded')
            # self.pos2_embedded = tf.nn.embedding_lookup(self.posEmbedding, self.pos2_ids, name='pos2_embedded')
            # self.sen_pos_embedded = tf.concat([self.sen_embedded, self.pos1_embedded, self.pos2_embedded], 2)

        def get_last(inputs, seq_len):
            batch_size = tf.shape(inputs)[0]
            max_seq_len = tf.shape(inputs)[1]
            dim = inputs.get_shape().as_list()[2]
            index_list = tf.range(batch_size) * max_seq_len + (seq_len - 1)
            last_outputs = tf.gather(tf.reshape(inputs, (-1, dim)), index_list, name='last_outputs')
            return last_outputs

        # # Bi-GRU layer. gain sentence vector
        # with tf.variable_scope('bigru'):
        #     fd_gru_cell = tf.nn.rnn_cell.GRUCell(gru_num_units)
        #     bd_gru_cell = tf.nn.rnn_cell.GRUCell(gru_num_units)
        #     if is_training:
        #         fd_gru_cell = tf.nn.rnn_cell.DropoutWrapper(fd_gru_cell, input_keep_prob=gru_input_keep_prob,
        #                                                     output_keep_prob=gru_output_keep_prob)
        #         bd_gru_cell = tf.nn.rnn_cell.DropoutWrapper(bd_gru_cell, input_keep_prob=gru_input_keep_prob,
        #                                                     output_keep_prob=gru_output_keep_prob)
        #     sen_outputs, sen_states = tf.nn.bidirectional_dynamic_rnn(fd_gru_cell, bd_gru_cell, self.sen_pos_embedded,
        #                                                               sequence_length=self.sen_len, dtype=tf.float32)
        #     sen_last_output = tf.maximum(sen_outputs[0], sen_outputs[1])

        # GRU Layer. gain entity vector
        # with tf.variable_scope('gru'):
        #     gru_cell = tf.nn.rnn_cell.GRUCell(gru_num_units)
        #     if is_training:
        #         gru_cell = tf.nn.rnn_cell.DropoutWrapper(gru_cell, input_keep_prob=gru_input_keep_prob,
        #                                                  output_keep_prob=gru_output_keep_prob)
        #     entity_outputs, entity_states = tf.nn.dynamic_rnn(gru_cell, self.entity_embedded,
        #                                                       sequence_length=self.entity_ids_len, dtype=tf.float32)
        #     entity_last_output = get_last(entity_outputs, self.entity_ids_len)

        # # Attention
        # entity_last_output = tf.reshape(entity_last_output, [-1, gru_num_units, 1])
        # alpha = tf.nn.softmax(tf.reshape(tf.matmul(sen_last_output, entity_last_output), [-1, sen_max_len]))
        # attention = tf.reshape(tf.matmul(sen_last_output, tf.reshape(alpha, [-1, sen_max_len, 1]), transpose_a=True),
        #                        [-1, gru_num_units])
        # if is_training:
        #     attention = tf.nn.dropout(attention, hidden_keep_prob)
        W1 = tf.get_variable("weight_1", shape=[300, hidden_dim], initializer=tf.truncated_normal_initializer())
        b1 = tf.get_variable("bias_1", shape=[hidden_dim], initializer=tf.zeros_initializer())
        output = tf.nn.relu(tf.nn.xw_plus_b(self.entity_vec, W1, b1))
        if is_training:
            output = tf.nn.dropout(output, hidden_keep_prob)

        W2 = tf.get_variable("weight_2", shape=[hidden_dim, n_class], initializer=tf.truncated_normal_initializer())
        b2 = tf.get_variable("bias_2", shape=[n_class], initializer=tf.zeros_initializer())
        logits = tf.nn.xw_plus_b(output, W2, b2)
        l2_loss = tf.nn.l2_loss(W2) + tf.nn.l2_loss(b2) + tf.nn.l2_loss(W1) + tf.nn.l2_loss(b1)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.labels)) \
                    + l2_loss_beta * l2_loss
        tf.summary.scalar('loss', self.loss)
        self.merged = tf.summary.merge_all()
        self.prob = tf.nn.softmax(logits)
        # 记录全局步数
        global_step = tf.Variable(0, trainable=False, name='global_step')
        # 选择使用的优化器
        if is_training:
            self.train_op = tf.train.GradientDescentOptimizer(starter_learning_rate).minimize(self.loss, global_step=global_step)


if __name__ == '__main__':
    print('load data........')
    with open('./../resource/newDictionary.pickle', 'rb') as f:
        save = pickle.load(f)
        wordEmbedding = save['wordEmbedding']
        sentences_vec = save['sentences_vec']
        sentences_id_len = save['sentences_id_len']
        pos_vec1 = save['pos_vec1']
        pos_vec2 = save['pos_vec2']
        entityPairs_vec = save['entityPairs_vec']
        entityPairs_len = save['entityPairs_len']
        labels = save['labels']
        del save
    with open('./../resource/vec.pickle', 'rb') as f:
        save = pickle.load(f)
        dataset = save['dataset']
        label = save['label']
        del save
    # 将label转为one-hot编码形式
    labels = reformat(np.array(labels))
    # 将vector由list形式转换为ndarray形式
    wordEmbedding = np.array(wordEmbedding)
    sentences_ids = sequence.pad_sequences(sentences_vec, maxlen=sen_max_len, truncating='post', padding='post')
    pos1_ids = sequence.pad_sequences(pos_vec1, maxlen=sen_max_len, truncating='post', padding='post', value=121)
    pos2_ids = sequence.pad_sequences(pos_vec2, maxlen=sen_max_len, truncating='post', padding='post', value=121)
    entity_ids = sequence.pad_sequences(entityPairs_vec, maxlen=entity_max_len, truncating='post', padding='post')

    # test_sentences_ids = sequence.pad_sequences(test_sentences_vec, maxlen=sen_max_len, truncating='post', padding='post')
    # 分割训练集和验证集
    train_sentences_ids = sentences_ids[157:, :]
    train_pos1_ids = pos1_ids[157:, :]
    train_pos2_ids = pos2_ids[157:, :]
    train_sentences_ids_len = sentences_id_len[157:]
    train_entity_ids = entity_ids[157:, :]
    train_entity_ids_len = entityPairs_len[157:]
    train_labels = labels[157:, :]

    train_dataset = dataset[157:, :]

    valid_sentences_ids = sentences_ids[:157, :]
    valid_pos1_ids = pos1_ids[:157, :]
    valid_pos2_ids = pos2_ids[:157, :]
    valid_sentences_ids_len = sentences_id_len[:157]
    valid_entity_ids = entity_ids[:157, :]
    valid_entity_ids_len = entityPairs_len[:157]
    valid_labels = labels[:157, :]

    valid_dataset = dataset[:157, :]

    print('Training set', train_sentences_ids.shape, train_pos1_ids.shape, train_pos2_ids.shape,
          train_sentences_ids_len.shape, train_entity_ids.shape, train_entity_ids_len.shape, train_labels.shape)

    with tf.Session() as sess:
        initializer = tf.truncated_normal_initializer()
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            train_model = Model(True, wordEmbedding)
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            eval_model = Model(False, wordEmbedding)

        summary_writer = tf.summary.FileWriter('./../resource/summary/', sess.graph)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=num_checkpoints)
        sess.run(tf.global_variables_initializer())
        # print(sess.run(train_model.posEmbedding))
        if is_train:
            max_acc = 0
            print('Start training.....')
            for step in range(num_steps):
                offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
                # batch_sentence_ids = train_sentences_ids[offset:(offset + batch_size), :]
                # batch_pos1_ids = train_pos1_ids[offset:(offset + batch_size), :]
                # batch_pos2_ids = train_pos2_ids[offset:(offset + batch_size), :]
                # batch_sentence_ids_len = train_sentences_ids_len[offset:(offset + batch_size)]
                # batch_entity_ids = train_entity_ids[offset:(offset + batch_size), :]
                # batch_entity_ids_len = train_entity_ids_len[offset:(offset + batch_size)]
                batch_labels = train_labels[offset:(offset + batch_size), :]

                batch_entity_vec = train_dataset[offset:(offset + batch_size), :]

                # feed_dict = {train_model.entity_ids: batch_entity_ids,
                #              train_model.entity_ids_len: batch_entity_ids_len,
                #              train_model.sen_ids: batch_sentence_ids,
                #              train_model.pos1_ids: batch_pos1_ids,
                #              train_model.pos2_ids: batch_pos2_ids,
                #              train_model.sen_len: batch_sentence_ids_len,
                #              train_model.labels: batch_labels}

                feed_dict = {train_model.entity_vec: batch_entity_vec,
                             train_model.labels: batch_labels}
                summary, _, loss, predictions = sess.run([train_model.merged, train_model.train_op, train_model.loss, train_model.prob], feed_dict=feed_dict)
                summary_writer.add_summary(summary, step)

                if step % evaluate_every == 0:
                    print("Minibatch loss at step %d: %f" % (step, loss))
                    print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
                    # feed_dic = {eval_model.entity_ids: valid_entity_ids,
                    #             eval_model.entity_ids_len: valid_entity_ids_len,
                    #             eval_model.sen_ids: valid_sentences_ids,
                    #             eval_model.pos1_ids: valid_pos1_ids,
                    #             eval_model.pos2_ids: valid_pos2_ids,
                    #             eval_model.sen_len: valid_sentences_ids_len,
                    #             eval_model.labels: valid_labels}
                    feed_dic = {eval_model.entity_vec: valid_dataset,
                                eval_model.labels: valid_labels}
                    valid_predictions = sess.run(eval_model.prob, feed_dict=feed_dic)
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
        # else:
        #     model_file = tf.train.latest_checkpoint('./../resource/model/')
        #     saver.restore(sess, model_file)
        #     feed_dic = {model.entity_vec: test_dataset,
        #                 model.sen_ids: test_sentences_ids,
        #                 model.sen_len: test_sentences_ids_len}
        #     test_predictions = sess.run(model.prob, feed_dict=feed_dic)
        #     test_result = np.argmax(test_predictions, 1)
        #     label = []
        #     for result in test_result:
        #         if result == 0 or result == 1:
        #             label.append("USAGE")
        #         elif result ==2 or result == 3:
        #             label.append("RESULT")
        #         elif result == 4 or result == 5:
        #             label.append("MODEL-FEATURE")
        #         elif result == 6 or result == 7:
        #             label.append("PART_WHOLE")
        #         elif result == 8 or result == 9:
        #             label.append("TOPIC")
        #         elif result == 10:
        #             label.append("COMPARE")
        #     final_result = []
        #     with open('./../resource/1.1.test.relations.txt', 'r') as f:
        #         stringList = f.readlines()
        #         for i, string_wyd in enumerate(stringList):
        #             final_result.append(label[i]+string_wyd)
        #     with open('./../resource/result.txt', 'w') as f:
        #         for result in final_result:
        #             f.write(result)