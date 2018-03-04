#!/usr/bin/python3
# -*- coding: UTF-8 -*-
'''
为了确定合理的分类数，这里使用elbow思想画图来确定分类，认为图中比较明确的拐点是合理的分类数，这中间使用了k-means聚类算法
另一个方法是bic Score算法，使用高斯聚类算法
'''
__author__ = 'gaoxianru'

from matplotlib import pyplot
from numpy import zeros, array, tile
from scipy.linalg import norm
import numpy.matlib as ml
import random
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import numpy as np
from sklearn import mixture
from sklearn import metrics
import csv
from sklearn.preprocessing import scale
from sklearn.mixture.gmm import GMM
import numpy as np

file_path = "./moredata/log0701-0708-0.csv"


# read data from csv
def get_data_withcsv(file_path):
    num_arr = []
    user_arr = []
    data_arr = []
    with open(file_path, 'rU') as f:
        reader = csv.reader(f)
        for row in reader:
            try:
                # scale(row[2:])
                num_arr.append(row[0])
                user_arr.append(row[1])
                data_arr.append(row[2:])
            except Exception as e:
                pass
    data_scaled = scale(data_arr)  # 对原始数据按特征维度进行标准化，即减去均值，除以方差，以消除不同维度特征之间的量纲差异
    # print(len(num_arr), len(user_arr), len(data_arr))
    print(data_scaled.mean(), data_scaled.std())
    return data_arr, data_scaled, user_arr


# read data from csv
data_raw, data_scaled, user_arr = get_data_withcsv(file_path)


def elbow(X):
    K = range(1, 10)  # 观察区间从1类到10类
    meandistortions = []
    for k in K:
        kmeans = KMeans(init='k-means++', n_clusters=k, random_state=1)
        kmeans.fit(X)
        meandistortions.append(sum(np.min(cdist(X, kmeans.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
    pyplot.plot(K, meandistortions, 'bx-')
    pyplot.xlabel('k')
    pyplot.ylabel('error')
    pyplot.title('elbow method')
    pyplot.show()


# elbow(data_scaled)


X = data_scaled

lowest_bic = np.infty
bic = []
n_components_range = range(2, 10)
cv_types = ['spherical', 'diag', 'full', 'tied']
for cv_type in cv_types:
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(n_components=n_components, covariance_type=cv_type, n_init=10)
        gmm.fit(X)
        bic.append(gmm.bic(X))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm
bic = np.array(bic)
# print(bic)

# draw the cluster gmm with bic score
import itertools
from matplotlib import pyplot as plt

n_components_range = range(2, 10)
bic = np.array(bic)
cv_types = ['spherical', 'diag', 'full', 'tied']
color_iter = itertools.cycle(['k', 'r', 'g', 'b', 'c', 'm', 'y'])
bars = []

# Plot the BIC scores
spl = plt.subplot(2, 1, 1)
for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
    xpos = np.array(n_components_range) + .2 * (i - 2)
    # print("xpos:",xpos)
    bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                  (i + 1) * len(n_components_range)],
                        width=.2, color=color))
plt.xticks(n_components_range)
plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
plt.title('BIC score per model')
xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 + \
       .2 * np.floor(bic.argmin() / len(n_components_range))
plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
spl.set_xlabel('Number of components')
spl.legend([b[0] for b in bars], cv_types)
plt.show()


gmm = mixture.GaussianMixture(n_components=4, covariance_type='diag', max_iter=1000, n_init=5)
gmm.fit(X)
pred = gmm.predict(X)

data_label = np.c_[pred, user_arr, data_raw]
c1, c2, c3, c4, c5, c6, c7, c8 = [], [], [], [], [], [], [], []

for data in data_label:
    if data[0] == '0':
        c1.append(data[1:])
    elif data[0] == '1':
        c2.append(data[1:])
    elif data[0] == '2':
        c3.append(data[1:])
    elif data[0] == '3':
        c4.append(data[1:])
    elif data[0] == '4':
        c5.append(data[1:])
    elif data[0] == '5':
        c6.append(data[1:])
    elif data[0] == '6':
        c7.append(data[1:])
    elif data[0] == '7':
        c8.append(data[1:])

mean = open("./moredata/mean.csv", 'a')
mean.truncate(0)
mean_writer = csv.writer(mean, dialect='excel')
mean_writer.writerow(['', 'room_per', 'fraud_per', 'social_score', 'prize_per', 'len'])
i = 1
for cx in (c1, c2, c3, c4, c5):
# for cx in (c1, c2, c3, c4, c5, c6, c7, c8):

    fileName = "./moredata/t%d.csv" % i
    out = open(fileName, 'a')
    out.truncate(0)
    csv_writer = csv.writer(out, dialect='excel')
    # csv_writer.writerow(['user_id', 'room_per', 'fraud_per', 'social_score', 'prize_per', 'label'])
    for c in cx:
        row = []
        row = np.append(row, c)
        row = np.append(row, [i-1])
        csv_writer.writerow(row)
    cn = np.array(cx)
    cn = cn[:, 1:]
    # print(cn)
    cn = cn.astype(float)
    cc = np.mean(cn, axis=0)
    print("c%d:" % i, len(cx), "mean:", cc)
    length = [len(cx)]
    row = ["class%d" % i, ]
    row = np.append(row, cc)
    row = np.append(row, length)
    mean_writer.writerow(row)
    i = i + 1
