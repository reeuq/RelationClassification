# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import tensorflow as tf
from keras.preprocessing import sequence
import sys
import os
from six.moves import cPickle as pickle

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# from evaluation.Evaluation import accuracy
# from evaluation.Evaluation import precision_each_class
# from evaluation.Evaluation import recall_each_class
# from evaluation.Evaluation import f1_each_class_precision_recall
from evaluation.Evaluation import f1_each_class
# from evaluation.Evaluation import class_label_count
# from evaluation.Evaluation import print_out

# 相对位置特征词典大小
pos_dic_max_len = 122
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
batch_size = 32
# 训练总步数
# num_steps = 5001
# 每训练多少批次，使用develop数据集验证
# evaluate_every = 100
# 存储模型个数
num_checkpoints = 5
# 初始学习率
starter_learning_rate = 0.1
# 训练轮数
epochs = 100
# 是否为训练阶段
is_train = True


def reformat(labels):
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(11) == labels[:, None]).astype(np.int32)
    return labels


class Model(object):
    def __init__(self, is_training, embedding_vector, scope):
        # placeholder
        self.entity_ids = tf.placeholder(tf.int32, [None, entity_max_len], name='entity_ids')
        self.entity_ids_len = tf.placeholder(tf.int32, [None], name='entity_len')
        self.sen_ids = tf.placeholder(tf.int32, [None, sen_max_len], name='sen_ids')
        self.pos1_ids = tf.placeholder(tf.int32, [None, sen_max_len], name='pos1_ids')
        self.pos2_ids = tf.placeholder(tf.int32, [None, sen_max_len], name='pos2_ids')
        self.sen_len = tf.placeholder(tf.int32, [None], name='sen_len')
        self.labels = tf.placeholder(tf.int32, [None, n_class], name='labels')

        # Embedding Layer
        # get embedding
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            embedding_matrix = tf.get_variable('embedding_matrix', shape=embedding_vector.shape, dtype=tf.float32,
                                               trainable=True, initializer=tf.constant_initializer(embedding_vector))
            self.sen_embedded = tf.nn.embedding_lookup(embedding_matrix, self.sen_ids, name='sen_embedded')
            self.entity_embedded = tf.nn.embedding_lookup(embedding_matrix, self.entity_ids, name='entity_embedded')
            self.posEmbedding_matrix = tf.get_variable('posEmbedding_matrix', shape=[pos_dic_max_len, pos_vec_dim], dtype=tf.float32,
                                                       initializer=tf.truncated_normal_initializer())
            self.pos1_embedded = tf.nn.embedding_lookup(self.posEmbedding_matrix, self.pos1_ids, name='pos1_embedded')
            self.pos2_embedded = tf.nn.embedding_lookup(self.posEmbedding_matrix, self.pos2_ids, name='pos2_embedded')
            self.sen_pos_embedded = tf.concat([self.sen_embedded, self.pos1_embedded, self.pos2_embedded], 2)

        def get_last(inputs, seq_len):
            batch_size = tf.shape(inputs)[0]
            max_seq_len = tf.shape(inputs)[1]
            dim = inputs.get_shape().as_list()[2]
            index_list = tf.range(batch_size) * max_seq_len + (seq_len - 1)
            last_outputs = tf.gather(tf.reshape(inputs, (-1, dim)), index_list, name='last_outputs')
            return last_outputs

        # # Bi-GRU layer. gain sentence vector
        with tf.variable_scope('bigru'):
            fd_gru_cell = tf.nn.rnn_cell.GRUCell(gru_num_units)
            bd_gru_cell = tf.nn.rnn_cell.GRUCell(gru_num_units)
            if is_training:
                fd_gru_cell = tf.nn.rnn_cell.DropoutWrapper(fd_gru_cell, input_keep_prob=gru_input_keep_prob,
                                                            output_keep_prob=gru_output_keep_prob)
                bd_gru_cell = tf.nn.rnn_cell.DropoutWrapper(bd_gru_cell, input_keep_prob=gru_input_keep_prob,
                                                            output_keep_prob=gru_output_keep_prob)
            sen_outputs, sen_states = tf.nn.bidirectional_dynamic_rnn(fd_gru_cell, bd_gru_cell, self.sen_pos_embedded,
                                                                      sequence_length=self.sen_len, dtype=tf.float32)
            sen_last_output = tf.maximum(sen_outputs[0], sen_outputs[1])

        # GRU Layer. gain entity vector
        with tf.variable_scope('gru'):
            gru_cell = tf.nn.rnn_cell.GRUCell(gru_num_units)
            if is_training:
                gru_cell = tf.nn.rnn_cell.DropoutWrapper(gru_cell, input_keep_prob=gru_input_keep_prob,
                                                         output_keep_prob=gru_output_keep_prob)
            entity_outputs, entity_states = tf.nn.dynamic_rnn(gru_cell, self.entity_embedded,
                                                              sequence_length=self.entity_ids_len, dtype=tf.float32)
            entity_last_output = get_last(entity_outputs, self.entity_ids_len)

        # Attention
        with tf.name_scope("attention"):
            entity_last_output = tf.reshape(entity_last_output, [-1, gru_num_units, 1])
            alpha = tf.nn.softmax(tf.reshape(tf.matmul(sen_last_output, entity_last_output), [-1, sen_max_len]))
            attention = tf.reshape(tf.matmul(sen_last_output, tf.reshape(alpha, [-1, sen_max_len, 1]),
                                             transpose_a=True),
                                   [-1, gru_num_units])
            if is_training:
                attention = tf.nn.dropout(attention, hidden_keep_prob)

        # W1 = tf.get_variable("weight_1", shape=[300, hidden_dim], initializer=tf.truncated_normal_initializer())
        # b1 = tf.get_variable("bias_1", shape=[hidden_dim], initializer=tf.zeros_initializer())
        # output = tf.nn.relu(tf.nn.xw_plus_b(self.entity_vec, W1, b1))
        # if is_training:
        #     output = tf.nn.dropout(output, hidden_keep_prob)

        with tf.name_scope("full_connect"):
            W2 = tf.get_variable("weight_2", shape=[gru_num_units, n_class],
                                 initializer=tf.truncated_normal_initializer())
            b2 = tf.get_variable("bias_2", shape=[n_class], initializer=tf.zeros_initializer())
            logits = tf.nn.xw_plus_b(attention, W2, b2)

        with tf.name_scope("loss"):
            l2_loss = tf.nn.l2_loss(W2) + tf.nn.l2_loss(b2)
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.labels)) + \
                l2_loss_beta * l2_loss
            tf.summary.scalar('loss', self.loss)

        with tf.name_scope("prediction"):
            self.prob = tf.nn.softmax(logits)

        with tf.name_scope("accuracy"):
            correct_pred = tf.equal(tf.argmax(self.prob, axis=1), tf.argmax(self.labels, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

        self.merged = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, scope))
        # 记录全局步数
        self.global_step = tf.get_variable('global_step', shape=[], trainable=False,
                                           initializer=tf.constant_initializer(0), dtype=tf.int32)
        # 选择使用的优化器
        if is_training:
            self.train_op = tf.train.GradientDescentOptimizer(starter_learning_rate)\
                .minimize(self.loss, global_step=self.global_step)


def get_batches(batch_size, *args):
    n_batches = (len(args[0])-1) // batch_size + 1
    new_args = []
    if len(args[0]) % n_batches != 0:
        for x in args:
            new_args.append(np.concatenate((x, x[:n_batches*batch_size - len(args[0])]), axis=0))
    else:
        new_args.extend(args)
    for i in range(0, len(new_args[0]), batch_size):
        data_batch = []
        for x in new_args:
            data_batch.append(x[i: i + batch_size])
        yield data_batch


# def make_predictions(lstm_size, multiple_fc, fc_units, checkpoint):
#     '''Predict the sentiment of the testing data'''
#
#     # Record all of the predictions
#     all_preds = []
#
#     model = build_rnn(n_words=n_words, embed_size=embed_size, batch_size=batch_size, lstm_size=lstm_size,
#                       num_layers=num_layers, dropout=dropout, learning_rate=learning_rate, multiple_fc=multiple_fc,
#                       fc_units=fc_units)
#
#     with tf.Session() as sess:
#         saver = tf.train.Saver()
#         # Load the model
#         saver.restore(sess, checkpoint)
#         test_state = sess.run(model.initial_state)
#         with tqdm(total=len(x_test)) as pbar:
#             for _, x in enumerate(get_test_batches(x_test, batch_size), 1):
#                 feed = {model.inputs: x, model.keep_prob: 1, model.initial_state: test_state}
#                 predic = sess.run([model.predictions], feed_dict=feed)
#                 pbar.update(batch_size)
#                 for pre in predic[0]:
#                     all_preds.append(round(float(pre)))
#
#     return all_preds


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

    valid_sentences_ids = sentences_ids[:157, :]
    valid_pos1_ids = pos1_ids[:157, :]
    valid_pos2_ids = pos2_ids[:157, :]
    valid_sentences_ids_len = sentences_id_len[:157]
    valid_entity_ids = entity_ids[:157, :]
    valid_entity_ids_len = entityPairs_len[:157]
    valid_labels = labels[:157, :]

    print('Training set', train_sentences_ids.shape, train_pos1_ids.shape, train_pos2_ids.shape,
          train_sentences_ids_len.shape, train_entity_ids.shape, train_entity_ids_len.shape, train_labels.shape)

    with tf.Session() as sess:
        with tf.name_scope("train") as train_scope:
            with tf.variable_scope("model", reuse=None):
                train_model = Model(True, wordEmbedding, train_scope)
        with tf.name_scope("test") as test_scope:
            with tf.variable_scope("model", reuse=True):
                eval_model = Model(False, wordEmbedding, test_scope)

        train_summary_writer = tf.summary.FileWriter('./../resource/summary/train', sess.graph)
        valid_summary_writer = tf.summary.FileWriter('./../resource/summary/valid')

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=num_checkpoints)
        sess.run(tf.global_variables_initializer())

        valid_loss_table = []
        early_stop = 0
        for e in range(epochs):
            train_loss = []
            train_acc = []
            train_avg_f1 = []

            for train_dataset in get_batches(batch_size, train_sentences_ids, train_pos1_ids, train_pos2_ids,
                                             train_sentences_ids_len, train_entity_ids, train_entity_ids_len,
                                             train_labels):
                batch_sentence_ids, batch_pos1_ids, batch_pos2_ids, batch_sentence_ids_len, batch_entity_ids, \
                    batch_entity_ids_len, batch_labels = train_dataset

                feed_dict = {train_model.entity_ids: batch_entity_ids,
                             train_model.entity_ids_len: batch_entity_ids_len,
                             train_model.sen_ids: batch_sentence_ids,
                             train_model.pos1_ids: batch_pos1_ids,
                             train_model.pos2_ids: batch_pos2_ids,
                             train_model.sen_len: batch_sentence_ids_len,
                             train_model.labels: batch_labels}

                train_accuracy, global_step, summary, _, loss, predictions = \
                    sess.run([train_model.accuracy, train_model.global_step, train_model.merged, train_model.train_op,
                              train_model.loss, train_model.prob], feed_dict=feed_dict)
                train_loss.append(loss)
                train_acc.append(train_accuracy * 100)
                train_avg_f1.append(np.mean(f1_each_class(predictions, batch_labels)))

                train_summary_writer.add_summary(summary, global_step)

            print("mean loss at epoch %d: %f" % (e, np.mean(train_loss)))
            print("mean accuracy: %.1f%%" % np.mean(train_acc))
            print("mean macro f1: %.1f%%" % np.mean(train_avg_f1))

            feed_dic = {eval_model.entity_ids: valid_entity_ids,
                        eval_model.entity_ids_len: valid_entity_ids_len,
                        eval_model.sen_ids: valid_sentences_ids,
                        eval_model.pos1_ids: valid_pos1_ids,
                        eval_model.pos2_ids: valid_pos2_ids,
                        eval_model.sen_len: valid_sentences_ids_len,
                        eval_model.labels: valid_labels}

            valid_accuracy, global_step, valid_summary, valid_loss, valid_predictions = \
                sess.run([eval_model.accuracy, eval_model.global_step, eval_model.merged, eval_model.loss,
                          eval_model.prob], feed_dict=feed_dic)

            valid_loss_table.append(valid_loss)
            valid_summary_writer.add_summary(valid_summary, global_step)
            predict_f1 = np.mean(f1_each_class(valid_predictions, valid_labels))

            if valid_loss > min(valid_loss_table):
                early_stop += 1
                if early_stop == 1000:
                    train_summary_writer.close()
                    valid_summary_writer.close()
                    break
            else:
                print("### New record ###")
                print("valid:loss={},acc={},f1={}".format(valid_loss, valid_accuracy, predict_f1))
                print()
                early_stop = 0  # 清零
                saver.save(sess, './../resource/model/classifier.ckpt')


















        # if is_train:
        #     max_acc = 0
        #     print('Start training.....')
        #     for step in range(num_steps):
        #         offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        #         # batch_sentence_ids = train_sentences_ids[offset:(offset + batch_size), :]
        #         # batch_pos1_ids = train_pos1_ids[offset:(offset + batch_size), :]
        #         # batch_pos2_ids = train_pos2_ids[offset:(offset + batch_size), :]
        #         # batch_sentence_ids_len = train_sentences_ids_len[offset:(offset + batch_size)]
        #         batch_entity_ids = train_entity_ids[offset:(offset + batch_size), :]
        #         batch_entity_ids_len = train_entity_ids_len[offset:(offset + batch_size)]
        #         batch_labels = train_labels[offset:(offset + batch_size), :]
        #
        #         # batch_entity_vec = train_dataset[offset:(offset + batch_size), :]
        #
        #         # feed_dict = {train_model.entity_ids: batch_entity_ids,
        #         #              train_model.entity_ids_len: batch_entity_ids_len,
        #         #              train_model.sen_ids: batch_sentence_ids,
        #         #              train_model.pos1_ids: batch_pos1_ids,
        #         #              train_model.pos2_ids: batch_pos2_ids,
        #         #              train_model.sen_len: batch_sentence_ids_len,
        #         #              train_model.labels: batch_labels}
        #
        #         feed_dict = {train_model.entity_ids: batch_entity_ids,
        #                      train_model.entity_ids_len: batch_entity_ids_len,
        #                      train_model.labels: batch_labels}
        #         summary, _, loss, predictions = sess.run([train_model.merged, train_model.train_op, train_model.loss,
        # train_model.prob], feed_dict=feed_dict)
        #         # summary_writer.add_summary(summary, step)
        #
        #         if step % evaluate_every == 0:
        #             print("Minibatch loss at step %d: %f" % (step, loss))
        #             print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
        #             # feed_dic = {eval_model.entity_ids: valid_entity_ids,
        #             #             eval_model.entity_ids_len: valid_entity_ids_len,
        #             #             eval_model.sen_ids: valid_sentences_ids,
        #             #             eval_model.pos1_ids: valid_pos1_ids,
        #             #             eval_model.pos2_ids: valid_pos2_ids,
        #             #             eval_model.sen_len: valid_sentences_ids_len,
        #             #             eval_model.labels: valid_labels}
        #             feed_dic = {eval_model.entity_ids: valid_entity_ids,
        #                         eval_model.entity_ids_len: valid_entity_ids_len,
        #                         eval_model.labels: valid_labels}
        #             valid_predictions = sess.run(eval_model.prob, feed_dict=feed_dic)
        #             predict_accuracy = accuracy(valid_predictions, valid_labels)
        #             if predict_accuracy > max_acc:
        #                 max_acc = predict_accuracy
        #                 saver.save(sess, './../resource/model/classifier.ckpt')
        #             print("Validation accuracy: %.1f%%" % predict_accuracy)
        #     print('---------------------------------------------')
        #     precision = precision_each_class(valid_predictions, valid_labels)
        #     recall = recall_each_class(valid_predictions, valid_labels)
        #     f1 = f1_each_class_precision_recall(precision, recall)
        #     count = class_label_count(valid_labels)
        #     print_out(precision, recall, f1, count)
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