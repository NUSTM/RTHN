# -*- encoding:utf-8 -*-
'''
@time: 2019/05/31
@author: mrzhang
@email: zhangmengran@njust.edu.cn
'''

import numpy as np
import tensorflow as tf
import os
import random
import pickle as pk
from sklearn.metrics import precision_score, recall_score, f1_score
path = '../data/'
max_doc_len = 75
max_sen_len = 45


def load_data():
    x = pk.load(open(path + 'x.txt', 'rb'))
    y = pk.load(open(path + 'y.txt', 'rb'))
    sen_len = pk.load(open(path + 'sen_len.txt', 'rb'))
    doc_len = pk.load(open(path + 'doc_len.txt', 'rb'))
    relative_pos = pk.load(open(path + 'relative_pos.txt', 'rb'))
    embedding = pk.load(open(path + 'embedding.txt', 'rb'))
    embedding_pos = pk.load(open(path + 'embedding_pos.txt', 'rb'))
    print('x.shape {} \ny.shape {} \nsen_len.shape {} \ndoc_len.shape {}\nrelative_pos.shape {}\nembedding_pos.shape {}'.format(x.shape, y.shape, sen_len.shape, doc_len.shape, relative_pos.shape, embedding_pos.shape))
    return x, y, sen_len, doc_len, relative_pos, embedding, embedding_pos


def acc_prf(pred_y, true_y, doc_len):
    tmp1, tmp2 = [], []
    for i in range(pred_y.shape[0]):
        for j in range(doc_len[i]):
            tmp1.append(pred_y[i][j])
            tmp2.append(true_y[i][j])
    y_pred, y_true = np.array(tmp1), np.array(tmp2)
    acc = precision_score(y_true, y_pred, average='micro')
    p = precision_score(y_true, y_pred, average='binary')
    r = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    return acc, p, r, f1


def batch_index(length, batch_size, test=False):
    index = list(range(length))
    if not test:
        random.shuffle(index)
    for i in range(int((length + batch_size - 1) / batch_size)):
        ret = index[i * batch_size: (i + 1) * batch_size]
        if not test and len(ret) < batch_size:
            break
        yield ret


def get_weight_varible(name, shape):
    return tf.get_variable(name, initializer=tf.random_uniform(shape, -0.01, 0.01))


def getmask(length, max_len, out_shape):
    '''
    length shape:[batch_size]
    '''
    # 转换成 0 1
    ret = tf.cast(tf.sequence_mask(length, max_len), tf.float32)
    return tf.reshape(ret, out_shape)


def biLSTM(inputs, length, n_hidden, scope):
    '''
    input shape:[batch_size, max_len, embedding_dim]
    length shape:[batch_size]
    return shape:[batch_size, max_len, n_hidden*2]
    '''
    outputs, state = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=tf.contrib.rnn.LSTMCell(n_hidden),
        cell_bw=tf.contrib.rnn.LSTMCell(n_hidden),
        inputs=inputs,
        sequence_length=length,
        dtype=tf.float32,
        scope=scope
    )

    return tf.concat(outputs, 2)


def softmax_by_length(inputs, length):
    '''
    input shape:[batch_size, 1, max_len]
    length shape:[batch_size]
    return shape:[batch_size, 1, max_len]
    '''
    inputs = tf.exp(tf.cast(inputs, tf.float32))
    inputs *= getmask(length, tf.shape(inputs)[2], tf.shape(inputs))
    _sum = tf.reduce_sum(inputs, reduction_indices=2, keep_dims=True) + 1e-9
    return inputs / _sum


def att_var(inputs, length, w1, b1, w2):
    '''
    input shape:[batch_size*max_doc_len, max_sen_len, n_hidden]
    length shape:[batch_size]
    return shape:[batch_size, n_hidden]
    '''
    max_len, n_hidden = (tf.shape(inputs)[1], tf.shape(inputs)[2])  # (45, n_hidden)
    # (batch_size*max_doc_len*max_sen_len, n_hidden)
    tmp = tf.reshape(inputs, [-1, n_hidden])
    u = tf.tanh(tf.matmul(tmp, w1) + b1)
    alpha = tf.reshape(tf.matmul(u, w2), [-1, 1, max_len])
    alpha = softmax_by_length(alpha, length)
    return tf.reshape(tf.matmul(alpha, inputs), [-1, n_hidden])


def maxS(alist):
    maxScore = 0.0
    maxIndex = -1
    for i in range(len(alist)):
        if alist[i] > maxScore:
            maxScore = alist[i]
            maxIndex = i
    return maxScore, maxIndex


def fun1(prob_pred, doc_len):
    ret = []
    for i in range(len(prob_pred)):
        ret.extend(list(prob_pred[i][:doc_len[i]]))
    return np.array(ret)

