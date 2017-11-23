from __future__ import print_function

import numpy as np


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


def precision_each_class(predictions, labels):
    zeros = np.zeros_like(labels)
    zeros[:, 0] = 1
    usage_reverse = (100.0 * np.sum((np.argmax(predictions, 1) == np.argmax(labels, 1))
                                    & (np.argmax(labels, 1) == np.argmax(zeros, 1)))
                     / np.sum(np.argmax(predictions, 1) == np.argmax(zeros, 1)))

    ones = np.zeros_like(labels)
    ones[:, 1] = 1
    usage = (100.0 * np.sum((np.argmax(predictions, 1) == np.argmax(labels, 1))
                            & (np.argmax(labels, 1) == np.argmax(ones, 1)))
             / np.sum(np.argmax(predictions, 1) == np.argmax(ones, 1)))

    twos = np.zeros_like(labels)
    twos[:, 2] = 1
    result_reverse = (100.0 * np.sum((np.argmax(predictions, 1) == np.argmax(labels, 1))
                                     & (np.argmax(labels, 1) == np.argmax(twos, 1)))
                      / np.sum(np.argmax(predictions, 1) == np.argmax(twos, 1)))

    threes = np.zeros_like(labels)
    threes[:, 3] = 1
    result = (100.0 * np.sum((np.argmax(predictions, 1) == np.argmax(labels, 1))
                             & (np.argmax(labels, 1) == np.argmax(threes, 1)))
              / np.sum(np.argmax(predictions, 1) == np.argmax(threes, 1)))

    fours = np.zeros_like(labels)
    fours[:, 4] = 1
    model_feature_reverse = (100.0 * np.sum((np.argmax(predictions, 1) == np.argmax(labels, 1))
                                            & (np.argmax(labels, 1) == np.argmax(fours, 1)))
                             / np.sum(np.argmax(predictions, 1) == np.argmax(fours, 1)))

    fives = np.zeros_like(labels)
    fives[:, 5] = 1
    model_feature = (100.0 * np.sum((np.argmax(predictions, 1) == np.argmax(labels, 1))
                                    & (np.argmax(labels, 1) == np.argmax(fives, 1)))
                     / np.sum(np.argmax(predictions, 1) == np.argmax(fives, 1)))

    sixes = np.zeros_like(labels)
    sixes[:, 6] = 1
    part_whole_reverse = (100.0 * np.sum((np.argmax(predictions, 1) == np.argmax(labels, 1))
                                         & (np.argmax(labels, 1) == np.argmax(sixes, 1)))
                          / np.sum(np.argmax(predictions, 1) == np.argmax(sixes, 1)))

    sevens = np.zeros_like(labels)
    sevens[:, 7] = 1
    part_whole = (100.0 * np.sum((np.argmax(predictions, 1) == np.argmax(labels, 1))
                                 & (np.argmax(labels, 1) == np.argmax(sevens, 1)))
                  / np.sum(np.argmax(predictions, 1) == np.argmax(sevens, 1)))

    eights = np.zeros_like(labels)
    eights[:, 8] = 1
    topic_reverse = (100.0 * np.sum((np.argmax(predictions, 1) == np.argmax(labels, 1))
                                    & (np.argmax(labels, 1) == np.argmax(eights, 1)))
                     / np.sum(np.argmax(predictions, 1) == np.argmax(eights, 1)))

    nines = np.zeros_like(labels)
    nines[:, 9] = 1
    topic = (100.0 * np.sum((np.argmax(predictions, 1) == np.argmax(labels, 1))
                            & (np.argmax(labels, 1) == np.argmax(nines, 1)))
             / np.sum(np.argmax(predictions, 1) == np.argmax(nines, 1)))

    tens = np.zeros_like(labels)
    tens[:, 10] = 1
    compare = (100.0 * np.sum((np.argmax(predictions, 1) == np.argmax(labels, 1))
                              & (np.argmax(labels, 1) == np.argmax(tens, 1)))
               / np.sum(np.argmax(predictions, 1) == np.argmax(tens, 1)))
    final_result = {
        "usage_reverse": usage_reverse,
        "usage": usage,
        "result_reverse": result_reverse,
        "result": result,
        "model_feature_reverse": model_feature_reverse,
        "model_feature": model_feature,
        "part_whole_reverse": part_whole_reverse,
        "part_whole": part_whole,
        "topic_reverse": topic_reverse,
        "topic": topic,
        "compare": compare
    }
    return final_result
