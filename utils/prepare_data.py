# -*- encoding:utf-8 -*-
'''
@time: 2019/05/31
@author: mrzhang
@email: zhangmengran@njust.edu.cn
'''

import codecs
import random
import numpy as np
import pickle as pk
from sklearn.metrics import precision_score, recall_score, f1_score
import pdb
path = '../data/'
max_doc_len = 75
max_sen_len = 45


def load_w2v(embedding_dim, embedding_dim_pos, train_file_path, embedding_path):
    print('\nload embedding...')
    words = []
    inputFile1 = codecs.open(train_file_path, 'r', 'utf-8')
    for line in inputFile1.readlines():
        line = line.strip().split(',')
        emotion, clause = line[2], line[-1]
        words.extend([emotion] + clause.split())
    inputFile1.close()
    words = set(words)  # 所有不重复词的集合
    word_idx = dict((c, k + 1) for k, c in enumerate(words))

    w2v = {}
    inputFile2 = codecs.open(embedding_path, 'r', 'utf-8')
    for line in inputFile2.readlines():
        line = line.strip().split(' ')
        w, ebd = line[0], line[1:]
        w2v[w] = ebd
    inputFile2.close()
    embedding = [list(np.zeros(embedding_dim))]

    hit = 0
    for item in words:
        if item in w2v:
            vec = list(map(float, w2v[item]))
            hit += 1
        else:
            vec = list(np.random.rand(embedding_dim) / 5. - 0.1)  # 从均匀分布[-0.1,0.1]中随机取
        embedding.append(vec)
    print('w2v_file: {}\nall_words: {} hit_words: {}'.format(
        embedding_path, len(words), hit))

    embedding_pos = [list(np.zeros(embedding_dim_pos))]
    embedding_pos.extend([list(np.random.normal(
        loc=0.0, scale=0.1, size=embedding_dim_pos)) for i in range(-68, 34)])
    # embedding.extend([list(np.random.normal(loc=0.0, scale=0.1, size=embedding_dim)) for i in range(-68,34)])
    embedding, embedding_pos = np.array(embedding), np.array(embedding_pos)
    pk.dump(embedding, open(path + 'embedding.txt', 'wb'))
    pk.dump(embedding_pos, open(path + 'embedding_pos.txt', 'wb'))

    print("embedding.shape: {} embedding_pos.shape: {}".format(
        embedding.shape, embedding_pos.shape))
    print("load embedding done!\n")
    return word_idx, embedding, embedding_pos


def load_data(input_file, word_idx, max_doc_len=max_doc_len, max_sen_len=max_sen_len):
    print('load data...')
    relative_pos, x, y, sen_len, doc_len = [], [], [], [], []

    y_clause_cause, clause_all, tmp_clause_len, relative_pos_all = np.zeros((max_doc_len, 2)), [], [], []
    next_ID = 2
    outputFile3 = codecs.open(input_file, 'r', 'utf-8')
    n_clause, yes_clause, no_clause, n_cut = [0] * 4

    for index, line in enumerate(outputFile3.readlines()):
        n_clause += 1
        line = line.strip().split(',')
        senID, clause_idx, emo_word, sen_pos, cause, words = int(line[0]), int(line[1]), line[2], int(line[3]), line[4], line[5]
        word_pos = sen_pos + 69
        if next_ID == senID:  # 数据文件末尾加了一个冗余的文档，会被丢弃
            doc_len.append(len(clause_all))

            for j in range(max_doc_len - len(clause_all)):
                clause_all.append(np.zeros((max_sen_len,)))
                tmp_clause_len.append(0)
                relative_pos_all.append(np.zeros((max_sen_len,)))
            relative_pos.append(relative_pos_all)
            x.append(clause_all)
            y.append(y_clause_cause)
            sen_len.append(tmp_clause_len)
            y_clause_cause, clause_all, tmp_clause_len, relative_pos_all = np.zeros((max_doc_len, 2)), [], [], []
            next_ID = senID + 1

        clause = [0] * max_sen_len
        relative_pos_clause = [0] * max_sen_len
        for i, word in enumerate(words.split()):
            clause[i] = int(word_idx[word])
            relative_pos_clause[i] = word_pos
        relative_pos_all.append(np.array(relative_pos_clause))
        clause_all.append(np.array(clause))
        tmp_clause_len.append(len(words.split()))
        if cause == 'no':
            no_clause += 1
            y_clause_cause[clause_idx - 1] = [1, 0]
        else:
            yes_clause += 1
            y_clause_cause[clause_idx - 1] = [0, 1]

    outputFile3.close()
    relative_pos, x, y, sen_len, doc_len = map(np.array, [relative_pos, x, y, sen_len, doc_len])

    pk.dump(relative_pos, open(path + 'relative_pos.txt', 'wb'))
    pk.dump(x, open(path + 'x.txt', 'wb'))
    pk.dump(y, open(path + 'y.txt', 'wb'))
    pk.dump(sen_len, open(path + 'sen_len.txt', 'wb'))
    pk.dump(doc_len, open(path + 'doc_len.txt', 'wb'))

    print('relative_pos.shape {}\nx.shape {} \ny.shape {} \nsen_len.shape {} \ndoc_len.shape {}\n'.format(
        relative_pos.shape, x.shape, y.shape, sen_len.shape, doc_len.shape
    ))
    print('n_clause {}, yes_clause {}, no_clause {}, n_cut {}'.format(n_clause, yes_clause, no_clause, n_cut))
    print('load data done!\n')
    return x, y, sen_len, doc_len


word_dict, _, _ = load_w2v(200, 50, path + 'clause_keywords.csv', path + 'w2v_200.txt')
load_data(path + 'clause_keywords.csv', word_dict)
