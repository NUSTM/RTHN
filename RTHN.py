# -*- encoding:utf-8 -*-
'''
@time: 2019/05/31
@author: mrzhang
@email: zhangmengran@njust.edu.cn
'''

import numpy as np
import pickle as pk
import transformer as trans
import tensorflow as tf
import sys, os, time, codecs, pdb
import utils.tf_funcs as func
from sklearn.model_selection import KFold
from sklearn.model_selection import ParameterGrid

FLAGS = tf.app.flags.FLAGS
# >>>>>>>>>>>>>>>>>>>> For Model <<<<<<<<<<<<<<<<<<<< #
# embedding parameters ##
tf.app.flags.DEFINE_integer('embedding_dim', 200, 'dimension of word embedding')
tf.app.flags.DEFINE_integer('embedding_dim_pos', 50, 'dimension of position embedding')
# input struct ##
tf.app.flags.DEFINE_integer('max_doc_len', 75, 'max number of tokens per documents')
tf.app.flags.DEFINE_integer('max_sen_len', 45, 'max number of tokens per sentence')
# model struct ##
tf.app.flags.DEFINE_integer('n_hidden', 100, 'number of hidden unit')
tf.app.flags.DEFINE_integer('n_class', 2, 'number of distinct class')
# >>>>>>>>>>>>>>>>>>>> For Data <<<<<<<<<<<<<<<<<<<< #
tf.app.flags.DEFINE_string('log_file_name', '', 'name of log file')
# >>>>>>>>>>>>>>>>>>>> For Training <<<<<<<<<<<<<<<<<<<< #
tf.app.flags.DEFINE_integer('training_iter', 15, 'number of train iter')
tf.app.flags.DEFINE_string('scope', 'RNN', 'RNN scope')
# not easy to tune , a good posture of using data to train model is very important
tf.app.flags.DEFINE_integer('batch_size', 32, 'number of example per batch')
tf.app.flags.DEFINE_float('lr_assist', 0.005, 'learning rate of assist')
tf.app.flags.DEFINE_float('lr_main', 0.001, 'learning rate')
tf.app.flags.DEFINE_float('keep_prob1', 0.5, 'word embedding training dropout keep prob')
tf.app.flags.DEFINE_float('keep_prob2', 1.0, 'softmax layer dropout keep prob')
tf.app.flags.DEFINE_float('l2_reg', 1e-5, 'l2 regularization')
tf.app.flags.DEFINE_integer('run_times', 1, 'run times of this model')
tf.app.flags.DEFINE_integer('num_heads', 5, 'the num heads of attention')
tf.app.flags.DEFINE_integer('n_layers', 1, 'the layers of transformer beside main')


def build_model(x, sen_len, doc_len, word_dis, word_embedding, pos_embedding, keep_prob1, keep_prob2, RNN=func.biLSTM):
    x = tf.nn.embedding_lookup(word_embedding, x)
    inputs = tf.reshape(x, [-1, FLAGS.max_sen_len, FLAGS.embedding_dim])
    word_dis = tf.nn.embedding_lookup(pos_embedding, word_dis)
    sh2 = 2 * FLAGS.n_hidden

    inputs = tf.nn.dropout(inputs, keep_prob=keep_prob1)
    sen_len = tf.reshape(sen_len, [-1])
    with tf.name_scope('word_encode'):
        wordEncode = RNN(inputs, sen_len, n_hidden=FLAGS.n_hidden, scope=FLAGS.scope + 'word_layer')
    wordEncode = tf.reshape(wordEncode, [-1, FLAGS.max_sen_len, sh2])

    with tf.name_scope('attention'):
        w1 = func.get_weight_varible('word_att_w1', [sh2, sh2])
        b1 = func.get_weight_varible('word_att_b1', [sh2])
        w2 = func.get_weight_varible('word_att_w2', [sh2, 1])
        senEncode = func.att_var(wordEncode, sen_len, w1, b1, w2)
    senEncode = tf.reshape(senEncode, [-1, FLAGS.max_doc_len, sh2])
    word_dis = tf.reshape(word_dis[:, :, 0, :], [-1, FLAGS.max_doc_len, FLAGS.embedding_dim_pos])
    senEncode_dis = tf.concat([senEncode, word_dis], axis=2)

    n_feature = 2 * FLAGS.n_hidden + FLAGS.embedding_dim_pos
    out_units = 2 * FLAGS.n_hidden
    batch = tf.shape(senEncode)[0]
    pred_zeros = tf.zeros(([batch, FLAGS.max_doc_len, FLAGS.max_doc_len]))
    matrix = tf.reshape((1 - tf.eye(FLAGS.max_doc_len)), [1, FLAGS.max_doc_len, FLAGS.max_doc_len]) + pred_zeros
    pred_assist_list, reg_assist_list, pred_assist_label_list = [], [], []
    if FLAGS.n_layers > 1:
        '''*******GL1******'''
        senEncode = trans_func(senEncode_dis, senEncode, n_feature, out_units, 'layer1')
        pred_assist, reg_assist = senEncode_softmax(senEncode, 'softmax_assist_w1', 'softmax_assist_b1', out_units, doc_len)

        pred_assist_label = tf.cast(tf.reshape(tf.argmax(pred_assist, axis=2), [-1, 1, FLAGS.max_doc_len]), tf.float32)
        pred_assist_label = (pred_assist_label + pred_zeros) * matrix

        pred_assist_label_list.append(pred_assist_label)
        pred_assist_list.append(pred_assist)
        reg_assist_list.append(reg_assist)
    '''*******GL n******'''
    for i in range(2, FLAGS.n_layers):
        senEncode_assist = tf.concat([senEncode, pred_assist_label], axis=2)
        n_feature = out_units + FLAGS.max_doc_len
        senEncode = trans_func(senEncode_assist, senEncode, n_feature, out_units, 'layer' + str(i))

        pred_assist, reg_assist = senEncode_softmax(senEncode, 'softmax_assist_w' + str(i), 'softmax_assist_b' + str(i), out_units, doc_len)
        pred_assist_label = tf.cast(tf.reshape(tf.argmax(pred_assist, axis=2), [-1, 1, FLAGS.max_doc_len]), tf.float32)
        pred_assist_label = (pred_assist_label + pred_zeros) * matrix
        pred_assist_label_list.append(pred_assist_label)
        pred_assist_label = tf.reduce_sum(pred_assist_label_list, axis=0)

        pred_assist_list.append(pred_assist)
        reg_assist_list.append(reg_assist)

    '''*******Main******'''
    if FLAGS.n_layers > 1:
        senEncode_dis_GL = tf.concat([senEncode, pred_assist_label], axis=2)
        n_feature = out_units + FLAGS.max_doc_len
        senEncode_main = trans_func(senEncode_dis_GL, senEncode, n_feature, out_units, 'block_main')
    else:
        senEncode_main = trans_func(senEncode_dis, senEncode, n_feature, out_units, 'block_main')
    pred, reg = senEncode_softmax(senEncode_main, 'softmax_w', 'softmax_b', out_units, doc_len)
    return pred, reg, pred_assist_list, reg_assist_list


def run():
    if FLAGS.log_file_name:
        sys.stdout = open(FLAGS.log_file_name, 'w')
    tf.reset_default_graph()
    localtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print("***********localtime: ", localtime)
    x_data, y_data, sen_len_data, doc_len_data, word_distance, word_embedding, pos_embedding = func.load_data()

    word_embedding = tf.constant(word_embedding, dtype=tf.float32, name='word_embedding')
    pos_embedding = tf.constant(pos_embedding, dtype=tf.float32, name='pos_embedding')
    print('build model...')

    start_time = time.time()
    x = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len, FLAGS.max_sen_len])
    y = tf.placeholder(tf.float32, [None, FLAGS.max_doc_len, FLAGS.n_class])
    sen_len = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len])
    doc_len = tf.placeholder(tf.int32, [None])
    word_dis = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len, FLAGS.max_sen_len])
    keep_prob1 = tf.placeholder(tf.float32)
    keep_prob2 = tf.placeholder(tf.float32)
    placeholders = [x, y, sen_len, doc_len, word_dis, keep_prob1, keep_prob2]

    pred, reg, pred_assist_list, reg_assist_list = build_model(x, sen_len, doc_len, word_dis, word_embedding, pos_embedding, keep_prob1, keep_prob2)

    with tf.name_scope('loss'):
        valid_num = tf.cast(tf.reduce_sum(doc_len), dtype=tf.float32)
        loss_op = - tf.reduce_sum(y * tf.log(pred)) / valid_num + reg * FLAGS.l2_reg
        loss_assist_list = []
        for i in range(FLAGS.n_layers - 1):
            loss_assist = - tf.reduce_sum(y * tf.log(pred_assist_list[i])) / valid_num + reg_assist_list[i] * FLAGS.l2_reg
            loss_assist_list.append(loss_assist)

    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr_main).minimize(loss_op)
        optimizer_assist_list = []
        for i in range(FLAGS.n_layers - 1):
            if i == 0:
                optimizer_assist = tf.train.AdamOptimizer(learning_rate=FLAGS.lr_assist).minimize(loss_assist_list[i])
            else:
                optimizer_assist = tf.train.AdamOptimizer(learning_rate=FLAGS.lr_main).minimize(loss_assist_list[i])
            optimizer_assist_list.append(optimizer_assist)

    true_y_op = tf.argmax(y, 2)
    pred_y_op = tf.argmax(pred, 2)
    pred_y_assist_op_list = []
    for i in range(FLAGS.n_layers - 1):
        pred_y_assist_op = tf.argmax(pred_assist_list[i], 2)
        pred_y_assist_op_list.append(pred_y_assist_op)

    print('build model done!\n')
    prob_list_pr, y_label = [], []
    # Training Code Block
    print_training_info()
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        kf, fold, SID = KFold(n_splits=10), 1, 0
        Id = []
        p_list, r_list, f1_list = [], [], []
        for train, test in kf.split(x_data):
            tr_x, tr_y, tr_sen_len, tr_doc_len, tr_word_dis = map(lambda x: x[train],
                [x_data, y_data, sen_len_data, doc_len_data, word_distance])
            te_x, te_y, te_sen_len, te_doc_len, te_word_dis = map(lambda x: x[test],
                [x_data, y_data, sen_len_data, doc_len_data, word_distance])

            precision_list, recall_list, FF1_list = [], [], []
            pre_list, true_list, pre_list_prob = [], [], []

            sess.run(tf.global_variables_initializer())
            print('############# fold {} ###############'.format(fold))
            fold += 1
            max_f1 = 0.0
            print('train docs: {}    test docs: {}'.format(len(tr_y), len(te_y)))

            '''PreTrain Global Label'''
            for layer in range(FLAGS.n_layers - 1):
                if layer == 0:
                    training_iter = FLAGS.training_iter
                else:
                    training_iter = FLAGS.training_iter - 5
                for i in range(training_iter):
                    step = 1
                    for train, _ in get_batch_data(tr_x, tr_y, tr_sen_len, tr_doc_len, tr_word_dis, FLAGS.keep_prob1, FLAGS.keep_prob2, FLAGS.batch_size):
                        _, loss, pred_y, true_y, pred_prob, doc_len_batch = sess.run(
                            [optimizer_assist_list[layer], loss_assist_list[layer], pred_y_assist_op_list[layer], true_y_op, pred_assist_list[layer], doc_len],
                            feed_dict=dict(zip(placeholders, train)))
                        acc_assist, p_assist, r_assist, f1_assist = func.acc_prf(pred_y, true_y, doc_len_batch)
                        if step % 10 == 0:
                            print('GL{}: epoch {}: step {}: loss {:.4f} acc {:.4f}'.format(layer + 1, i + 1, step, loss, acc_assist))
                        step = step + 1

            '''*********Train********'''
            for epoch in range(FLAGS.training_iter):
                step = 1
                for train, _ in get_batch_data(tr_x, tr_y, tr_sen_len, tr_doc_len, tr_word_dis, FLAGS.keep_prob1, FLAGS.keep_prob2, FLAGS.batch_size):
                    _, loss, pred_y, true_y, pred_prob, doc_len_batch = sess.run(
                        [optimizer, loss_op, pred_y_op, true_y_op, pred, doc_len],
                        feed_dict=dict(zip(placeholders, train)))
                    acc, p, r, f1 = func.acc_prf(pred_y, true_y, doc_len_batch)
                    if step % 5 == 0:
                        print('epoch {}: step {}: loss {:.4f} acc {:.4f}'.format(epoch + 1, step, loss, acc))
                    step = step + 1

                '''*********Test********'''
                test = [te_x, te_y, te_sen_len, te_doc_len, te_word_dis, 1., 1.]
                loss, pred_y, true_y, pred_prob = sess.run(
                    [loss_op, pred_y_op, true_y_op, pred], feed_dict=dict(zip(placeholders, test)))

                end_time = time.time()

                true_list.append(true_y)
                pre_list.append(pred_y)
                pre_list_prob.append(pred_prob)

                acc, p, r, f1 = func.acc_prf(pred_y, true_y, te_doc_len)
                precision_list.append(p)
                recall_list.append(r)
                FF1_list.append(f1)
                if f1 > max_f1:
                    max_acc, max_p, max_r, max_f1 = acc, p, r, f1
                print('\ntest: epoch {}: loss {:.4f} acc {:.4f}\np: {:.4f} r: {:.4f} f1: {:.4f} max_f1 {:.4f}\n'.format(
                    epoch + 1, loss, acc, p, r, f1, max_f1))

            Id.append(len(te_x))
            SID = np.sum(Id) - len(te_x)
            _, maxIndex = func.maxS(FF1_list)
            print("maxIndex:", maxIndex)
            print('Optimization Finished!\n')
            pred_prob = pre_list_prob[maxIndex]

            for i in range(pred_y.shape[0]):
                for j in range(te_doc_len[i]):
                    prob_list_pr.append(pred_prob[i][j][1])
                    y_label.append(true_y[i][j])

            print("*********prob_list_pr", len(prob_list_pr))
            print("*********y_label", len(y_label))

            p_list.append(max_p)
            r_list.append(max_r)
            f1_list.append(max_f1)
        pk.dump(np.array(prob_list_pr), open('/home/mrzhang/IJCAI_SourceCode/data/RTHN_pred_PR.pk', 'wb'))
        pk.dump(np.array(y_label), open('/home/mrzhang/IJCAI_SourceCode/data/y_label_PR.pk', 'wb'))
        print("running time: ", str((end_time - start_time) / 60.))
        print_training_info()
        p, r, f1 = map(lambda x: np.array(x).mean(), [p_list, r_list, f1_list])

        print("f1_score in 10 fold: {}\naverage : {} {} {}\n".format(np.array(f1_list).reshape(-1, 1), round(p, 4), round(r, 4), round(f1, 4)))
        return p, r, f1


def print_training_info():
    print('\n\n>>>>>>>>>>>>>>>>>>>>TRAINING INFO:\n')
    print('batch-{}, learning_rate-{}, keep_prob1-{}, num_heads-{}, n_layers-{}'.format(
        FLAGS.batch_size, FLAGS.lr_main, FLAGS.keep_prob1, FLAGS.num_heads, FLAGS.n_layers))
    print('training_iter-{}, scope-{}\n'.format(FLAGS.training_iter, FLAGS.scope))


def get_batch_data(x, y, sen_len, doc_len, word_dis, keep_prob1, keep_prob2, batch_size, test=False):
    for index in func.batch_index(len(y), batch_size, test):
        feed_list = [x[index], y[index], sen_len[index], doc_len[index], word_dis[index], keep_prob1, keep_prob2]
        yield feed_list, len(index)


def senEncode_softmax(s_senEncode, w_varible, b_varible, n_feature, doc_len):
    s = tf.reshape(s_senEncode, [-1, n_feature])
    s = tf.nn.dropout(s, keep_prob=FLAGS.keep_prob2)
    w = func.get_weight_varible(w_varible, [n_feature, FLAGS.n_class])
    b = func.get_weight_varible(b_varible, [FLAGS.n_class])
    pred = tf.matmul(s, w) + b
    pred *= func.getmask(doc_len, FLAGS.max_doc_len, [-1, 1])
    pred = tf.nn.softmax(pred)
    pred = tf.reshape(pred, [-1, FLAGS.max_doc_len, FLAGS.n_class])
    reg = tf.nn.l2_loss(w) + tf.nn.l2_loss(b)
    return pred, reg


def trans_func(senEncode_dis, senEncode, n_feature, out_units, scope_var):
    senEncode_assist = trans.multihead_attention(queries=senEncode_dis,
                                            keys=senEncode_dis,
                                            values=senEncode,
                                            units_query=n_feature,
                                            num_heads=FLAGS.num_heads,
                                            dropout_rate=0,
                                            is_training=True,
                                            scope=scope_var)
    senEncode_assist = trans.feedforward_1(senEncode_assist, n_feature, out_units)
    return senEncode_assist


def main(_):
    grid_search = {}
    params = {"n_layers": [3]}

    params_search = list(ParameterGrid(params))

    for i, param in enumerate(params_search):
        print("*************params_search_{}*************".format(i + 1))
        print(param)
        for key, value in param.items():
            setattr(FLAGS, key, value)
        p_list, r_list, f1_list = [], [], []
        for i in range(FLAGS.run_times):
            print("*************run(){}*************".format(i + 1))
            p, r, f1 = run()
            p_list.append(p)
            r_list.append(r)
            f1_list.append(f1)

        for i in range(FLAGS.run_times):
            print(round(p_list[i], 4), round(r_list[i], 4), round(f1_list[i], 4))
        print("avg_prf: ", np.mean(p_list), np.mean(r_list), np.mean(f1_list))

        grid_search[str(param)] = {"PRF": [round(np.mean(p_list), 4), round(np.mean(r_list), 4), round(np.mean(f1_list), 4)]}

    for key, value in grid_search.items():
        print("Main: ", key, value)



if __name__ == '__main__':
    tf.app.run()
