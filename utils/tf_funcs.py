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
path = '/home/mrzhang/IJCAI_SourceCode/data/'
# path = '/home/mrzhang/1080Ti/data/'
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


def cirCorrlation(H, Ht, n_hidden):
    Ht = tf.reshape(Ht, [-1, max_doc_len, max_sen_len, n_hidden])[:, :, 0, :]
    Ht = tf.reshape(Ht, [-1, n_hidden])   # [batch*max_doc, n_hidden]
    for i in range(n_hidden):
        tt = tf.reshape(Ht, [-1, n_hidden, 1])
        if i == 0:
            fusion = tf.matmul(H, tt)
        else:
            fusion = tf.concat([fusion, tf.matmul(H, tt)], 2)
        Ht = tf.manip.roll(Ht, shift=[-1], axis=[1])
    # fusion = tf.concat([fusion, H], 2)  # fusion:[-1, max_sen_len, 200] --> [-1, max_sen_len, 400]
    print("fusion", fusion)
    return fusion


def cirConvolution(H, Ht, n_hidden):
    '''
    emotion: Ht [-1, max_sen_len, 2*n_hidden]
    wordEnocde : H [-1, max_sen_len, 2*n_hidden]
    '''
    Ht = tf.reshape(tf.reshape(Ht, [-1, max_doc_len, max_sen_len, n_hidden])[:, :, 0, :], [-1, n_hidden])
    Ht = tf.reverse(Ht, axis=[-1])
    Ht = tf.manip.roll(Ht, shift=[1], axis=[-1])
    for i in range(n_hidden):
        tt = tf.reshape(Ht, [-1, n_hidden, 1])
        if i == 0:
            fusion = tf.matmul(H, tt)  # [-1, 45, 1]
        else:
            fusion = tf.concat([fusion, tf.matmul(H, tt)], 2)
        Ht = tf.manip.roll(Ht, shift=[1], axis=[-1])
    # # fusion = tf.concat([fusion, H], 2)  # fusion:[-1, max_sen_len, 200] --> [-1, max_sen_len, 400]
    fusion = tf.reshape(fusion, [-1, max_sen_len, n_hidden])
    return fusion


def fusion_att(wordEncode, wordEncode_emo, length, w1, w2, w3, w4):
    # y = tanh(W M)
    # a = softmax(w y)
    # r = H a
    # r = tanh(w r + w h)
    '''
    wordEncode shape:[batch_size*max_doc_len, max_sen_len, n_hidden]
    length shape:[batch_size]
    return shape:[batch_size, n_hidden]
    '''
    max_len, n_hidden = (tf.shape(wordEncode)[1], tf.shape(wordEncode)[2])  # (45, n_hidden)
    # # (batch_size*max_doc_len*max_sen_len, n_hidden)
    wordEncode_emo = tf.reshape(wordEncode_emo, [-1, n_hidden])
    u = tf.tanh(tf.matmul(wordEncode_emo, w1))
    alpha = tf.reshape(tf.matmul(u, w2), [-1, 1, max_len])
    alpha = softmax_by_length(alpha, length)
    r = tf.reshape(tf.matmul(alpha, wordEncode), [-1, n_hidden])

    # batch_size = tf.shape(wordEncode)[0]
    # max_len = tf.shape(wordEncode)[1]
    # index = tf.range(0, batch_size) * max_len + tf.maximum((length - 1), 0)
    # last = tf.gather(tf.reshape(wordEncode, [-1, n_hidden]), index)
    # r = tf.tanh(tf.matmul(last, w3) + tf.matmul(r, w4))
    return r


class Saver(object):
    def __init__(self, sess, save_dir, max_to_keep=10):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.sess = sess
        self.save_dir = save_dir
        self.saver = tf.train.Saver(
            write_version=tf.train.SaverDef.V2, max_to_keep=max_to_keep)

    def save(self, step):
        self.saver.save(self.sess, self.save_dir, global_step=step)

    def restore(self, idx=''):
        ckpt = tf.train.get_checkpoint_state(self.save_dir)
        model_path = self.save_dir + idx if idx else ckpt.model_checkpoint_path  # 'dir/-110'
        print("Reading model parameters from %s" % model_path)
        self.saver.restore(self.sess, model_path)


def get_weight_varible(name, shape):
    return tf.get_variable(name, initializer=tf.random_uniform(shape, -0.01, 0.01))


def getmask(length, max_len, out_shape):
    '''
    length shape:[batch_size]
    '''
    # 转换成 0 1
    ret = tf.cast(tf.sequence_mask(length, max_len), tf.float32)
    return tf.reshape(ret, out_shape)

# 实际运行比biLSTM更快


def biLSTM_multigpu(inputs, length, n_hidden, scope):
    '''
    input shape:[batch_size, max_len, embedding_dim]
    length shape:[batch_size]
    return shape:[batch_size, max_len, n_hidden*2]
    '''
    outputs, state = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=tf.contrib.rnn.LSTMCell(n_hidden),
        cell_bw=tf.contrib.rnn.LSTMCell(n_hidden),
        inputs=inputs,
        # sequence_length=length,
        dtype=tf.float32,
        scope=scope
    )

    max_len = tf.shape(inputs)[1]
    mask = getmask(length, max_len, [-1, max_len, 1])
    return tf.concat(outputs, 2) * mask


def LSTM_multigpu(inputs, length, n_hidden, scope):
    '''
    input shape:[batch_size, max_len, embedding_dim]
    length shape:[batch_size]
    return shape:[batch_size, max_len, n_hidden*2]
    '''
    outputs, state = tf.nn.dynamic_rnn(
        cell=tf.contrib.rnn.LSTMCell(n_hidden),
        inputs=inputs,
        # sequence_length=length,
        dtype=tf.float32,
        scope=scope
    )

    max_len = tf.shape(inputs)[1]
    mask = getmask(length, max_len, [-1, max_len, 1])
    return outputs * mask


def biLSTM_multigpu_last(inputs, length, n_hidden, scope):
    '''
    input shape:[batch_size, max_len, embedding_dim]
    length shape:[batch_size]
    return shape:[batch_size, max_len, n_hidden*2]
    '''
    outputs, state = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=tf.contrib.rnn.LSTMCell(n_hidden),
        cell_bw=tf.contrib.rnn.LSTMCell(n_hidden),
        inputs=inputs,
        # sequence_length=length,
        dtype=tf.float32,
        scope=scope
    )

    batch_size = tf.shape(inputs)[0]
    max_len = tf.shape(inputs)[1]

    index = tf.range(0, batch_size) * max_len + tf.maximum((length - 1), 0)
    # batch_size * n_hidden
    fw_last = tf.gather(tf.reshape(outputs[0], [-1, n_hidden]), index)
    index = tf.range(0, batch_size) * max_len
    # batch_size * n_hidden
    bw_last = tf.gather(tf.reshape(outputs[1], [-1, n_hidden]), index)

    return tf.concat([fw_last, bw_last], 1)


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


def LSTM(inputs, sequence_length, n_hidden, scope):
    outputs, state = tf.nn.dynamic_rnn(
        cell=tf.contrib.rnn.LSTMCell(n_hidden),
        inputs=inputs,
        sequence_length=sequence_length,
        dtype=tf.float32,
        scope=scope
    )
    return outputs


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


def att_avg(inputs, length):
    '''
    inputs shape:[batch_size, max_len, n_hidden]
    length shape:[batch_size]
    return shape:[batch_size, n_hidden]
    '''
    max_len = tf.shape(inputs)[1]
    # getmask(length, max_len, [-1, max_len, 1]) [batch_size, max_len, 1]
    inputs *= getmask(length, max_len, [-1, max_len, 1])
    inputs = tf.reduce_sum(inputs, 1, keep_dims=False)
    length = tf.cast(tf.reshape(length, [-1, 1]), tf.float32) + 1e-9
    return inputs / length


def penalty(pred, doc_len, max_doc_len, penlty_para):
    pred_prob1 = tf.reshape(pred[:, :, 1], [-1, max_doc_len])
    pred_prob1 *= getmask(doc_len, max_doc_len, [-1, max_doc_len])
    predmax_idx = tf.argmax(pred_prob1, 1)  # 每一个样本最大概率的索引值
    # 主任务
    a = (tf.ceil(tf.maximum(0., 0.5 - tf.reduce_max(pred_prob1, reduction_indices=1))) * penlty_para + 1)  # [0,0,0,1,0]*penlty + 1
    a = tf.reshape(a, [-1, 1, 1])
    return a, predmax_idx


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


def rectify(SID, pred_2d, doc_len_batch):
    # 如果预测全为0，最大概率那个设为1
    pred_1d = np.argmax(np.array(pred_2d), axis=2)
    for j in range(len(pred_1d)):
        SID += 1
        pred2d_sentence = pred_2d[j][:doc_len_batch[j]]
        pre = pred2d_sentence[:, 1]
        pred1d_sentence = pred_1d[j][:doc_len_batch[j]]
        if max(pred1d_sentence) == 0:
            index = np.argmax(pre)
            pred_2d[j][index] = [0, 1]

    # pred_y = np.argmax(pred_2d, axis=2)
    return SID, pred_2d


def rectify_sigmoid(SID, pred, doc_len_batch):
    # 如果预测全为0，最大概率那个设为1
    for j in range(len(pred)):
        SID += 1
        pred_sentence = pred[j][:doc_len_batch[j]]
        if max(pred_sentence) < 0.5:
            index = np.argmax(pred_sentence)
            pred[j][index] = 1.0
    return SID, pred


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = [g for g, _ in grad_and_vars]
        # Average over the 'tower' dimension.
        grad = tf.stack(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads
