# -*- coding: UTF-8 -*-
__author__='gaoxianru'

'''
简单神经网络预测
'''

import tensorflow as tf
import load_data
import numpy as np
import random
user_arr, data_arr, label = load_data.load_data()
data_label = np.c_[user_arr, data_arr, label]
# 打乱原始数据
np.random.shuffle(data_label)
size = len(user_arr)
test_len = int(size/4)  # 取1/4作为测试数据
test = data_label[:test_len]
train = data_label[test_len:]

test_usr = test[:, 0]
test_data = test[:, 1:7]
test_label = test[:, 7:]

sess = tf.InteractiveSession()

# 变量初始化
x = tf.placeholder(tf.float32, [None, 6])
w1 = tf.Variable(tf.truncated_normal([6, 8], stddev=0.1))
b1 = tf.Variable(tf.zeros([5]))
w2 = tf.Variable(tf.truncated_normal([8, 6], stddev=0.1))
b2 = tf.Variable(tf.zeros([5]))
w3 = tf.Variable(tf.truncated_normal([6, 4], stddev=0.1))
b3 = tf.Variable(tf.zeros([4]))
y_ = tf.placeholder(tf.float32, [None, 4])

# 定义假设韩式计算公式，TensorFlow会自动计算forward和backend方法
y1 = tf.nn.sigmoid(tf.matmul(x, w1) + b1)
y2 = tf.nn.sigmoid(tf.matmul(y1, w2)+b2)
yLogits = tf.matmul(y2, w3) + b3
y = tf.nn.softmax(yLogits)
# 定义损失函数公式，信息熵
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
# 定义优化算法SDG学习率0.5
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# 参数初始化
tf.global_variables_initializer().run()

train_len = len(train)
Epochs = 1000
# 迭代执行训练操作
for i in range(Epochs):  # 迭代1000次
    # 选全部样本计算量太大，只取一小部分进行随机梯度下降
    batch_size = 50
    mini_datas = []
    mini_labels = []
    np.random.shuffle(train)
    for k in range(0, train_len, batch_size):
        train_usr = train[k:k+batch_size, 0]
        train_data = train[k:k+batch_size, 1:7]
        train_label = train[k:k+batch_size, 7:]
        mini_datas.append(train_data)
        mini_labels.append(train_label)
    n = len(mini_labels)
    for j in range(n):
        sess.run(train_step, feed_dict={x: mini_datas[j], y_: mini_labels[j]})

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))  # 评估
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))  # 将结果转换成浮点数
    print("Epoch%d:%f" % (i+1, accuracy.eval({x: test_data, y_: test_label})))

sess.close()
# pred = sess.run(accuracy, feed_dict={x: test_data, y_: test_label})
# print(pred)
