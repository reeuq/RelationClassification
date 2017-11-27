from __future__ import print_function

import numpy as np


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


def precision_each_class(predictions, labels):
    categories = [-1] * 11
    for a in range(0, 11):
        zeros = np.zeros_like(labels)
        zeros[:, a] = 1
        categories[a] = (100.0 * np.sum((np.argmax(predictions, 1) == np.argmax(labels, 1))
                                        & (np.argmax(labels, 1) == np.argmax(zeros, 1)))
                         / np.sum(np.argmax(predictions, 1) == np.argmax(zeros, 1)))
    final_result = {
        "usage": ("%.2f" % categories[0]),
        "usage_reverse": ("%.2f" % categories[1]),
        "result": ("%.2f" % categories[2]),
        "result_reverse": ("%.2f" % categories[3]),
        "model_feature": ("%.2f" % categories[4]),
        "model_feature_reverse": ("%.2f" % categories[5]),
        "part_whole": ("%.2f" % categories[6]),
        "part_whole_reverse": ("%.2f" % categories[7]),
        "topic": ("%.2f" % categories[8]),
        "topic_reverse": ("%.2f" % categories[9]),
        "compare": ("%.2f" % categories[10])
    }
    return final_result


def recall_each_class(predictions, labels):
    categories = [-1] * 11
    for a in range(0, 11):
        zeros = np.zeros_like(labels)
        zeros[:, a] = 1
        categories[a] = (100.0 * np.sum((np.argmax(predictions, 1) == np.argmax(labels, 1))
                                        & (np.argmax(labels, 1) == np.argmax(zeros, 1)))
                         / np.sum(np.argmax(labels, 1) == np.argmax(zeros, 1)))
    final_result = {
        "usage": ("%.2f" % categories[0]),
        "usage_reverse": ("%.2f" % categories[1]),
        "result": ("%.2f" % categories[2]),
        "result_reverse": ("%.2f" % categories[3]),
        "model_feature": ("%.2f" % categories[4]),
        "model_feature_reverse": ("%.2f" % categories[5]),
        "part_whole": ("%.2f" % categories[6]),
        "part_whole_reverse": ("%.2f" % categories[7]),
        "topic": ("%.2f" % categories[8]),
        "topic_reverse": ("%.2f" % categories[9]),
        "compare": ("%.2f" % categories[10])
    }
    return final_result


def f1_each_class(predictions, labels):
    categories = [-1] * 11
    for a in range(0, 11):
        zeros = np.zeros_like(labels)
        zeros[:, a] = 1
        precision = (100.0 * np.sum((np.argmax(predictions, 1) == np.argmax(labels, 1))
                                    & (np.argmax(labels, 1) == np.argmax(zeros, 1)))
                     / np.sum(np.argmax(predictions, 1) == np.argmax(zeros, 1)))
        recall = (100.0 * np.sum((np.argmax(predictions, 1) == np.argmax(labels, 1))
                                 & (np.argmax(labels, 1) == np.argmax(zeros, 1)))
                  / np.sum(np.argmax(labels, 1) == np.argmax(zeros, 1)))
        categories[a] = (2 * precision * recall) / (precision + recall)
    final_result = {
        "usage": ("%.2f" % categories[0]),
        "usage_reverse": ("%.2f" % categories[1]),
        "result": ("%.2f" % categories[2]),
        "result_reverse": ("%.2f" % categories[3]),
        "model_feature": ("%.2f" % categories[4]),
        "model_feature_reverse": ("%.2f" % categories[5]),
        "part_whole": ("%.2f" % categories[6]),
        "part_whole_reverse": ("%.2f" % categories[7]),
        "topic": ("%.2f" % categories[8]),
        "topic_reverse": ("%.2f" % categories[9]),
        "compare": ("%.2f" % categories[10])
    }
    return final_result


def f1_each_class_precision_recall(precision, recall):
    final_result = {
        "usage": -1,
        "usage_reverse": -1,
        "result": -1,
        "result_reverse": -1,
        "model_feature": -1,
        "model_feature_reverse": -1,
        "part_whole": -1,
        "part_whole_reverse": -1,
        "topic": -1,
        "topic_reverse": -1,
        "compare": -1
    }
    for a in final_result.keys():
        temp = (2 * float(precision[a]) * float(recall[a])) / (float(precision[a]) + float(recall[a]))
        final_result[a] = ("%.2f" % temp)
    return final_result