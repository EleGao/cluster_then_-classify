# -*- coding: UTF-8 -*-
'''
读取csv文件中打好标签的分类数据，并对标签进行one-hot编码
'''
import numpy as np
import tensorflow as tf
import csv
from sklearn.preprocessing import scale

def load_data():
    # 用dict乱序excel里的数据
    dmap = {}
    i = 1
    for k in range(4):
        fileName = "moredata/6dim/t%d.csv" % (k+1)
        with open(fileName, 'rU') as f:
            reader = csv.reader(f)
            for row in reader:
                dmap[i] = row
                i = i+1

    usr_arr = []
    data_arr = []
    label_arr = []
    for (k, v) in dmap.items():
        usr_arr.append(v[0])
        data_arr.append(v[1:7])
        label_arr.append(v[7])

    data_scaled = scale(data_arr)
    one_hot_label = []
    CLASS = 4
    labels = np.array(label_arr)
    labels = labels.astype(int)
    b = tf.one_hot(labels, CLASS, 1, 0)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        one_hot_label = sess.run(b)

    return usr_arr, data_scaled, one_hot_label


