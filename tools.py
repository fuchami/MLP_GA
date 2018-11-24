# coding: utf-8

"""

いろいろの関数


"""

import keras.backend as K

""" f score define """
def precision(y_true, y_pred):
    # Calculates the precision
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

# 検出率　TP / (TP + FN)
def recall(y_true, y_pred):
    # Calculates the recall
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

# fscore
def f_score(y_true, y_pred, beta=1):
        if beta < 0:
                raise ValueError('The lowest choosable beta is zero (only precision).')
        
        if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
                return 0
        
        p = precision(y_true, y_pred)
        r = recall(y_true, y_pred)
        bb = beta ** 2
        f_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
        return f_score