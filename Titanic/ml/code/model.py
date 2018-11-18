#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @Time    : 2018/4/3 20:10
# @Author  : Spareribs
# @File    : base.py
"""

import pandas as pd  # 数据分析
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import tensorflow as tf

def logistic(X, y):
    """
    逻辑回归建模
    :param X:
    :param y:
    :return:
    """
    clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    # print(clf)
    clf.fit(X, y)
    return clf


def train_tree(dataset_X, dataset_Y):
    X_train, X_val, Y_train, Y_val = train_test_split(dataset_X.as_matrix(),
                                                      dataset_Y.as_matrix(),
                                                      test_size=0.2,
                                                      random_state=42)

    x = tf.placeholder(tf.float32, shape=[None, 6], name='input')
    y = tf.placeholder(tf.float32, shape=[None, 2], name='label')
    weights1 = tf.Variable(tf.random_normal([6, 6]), name='weights1')
    bias1 = tf.Variable(tf.zeros([6]), name='bias1')
    a = tf.nn.relu(tf.matmul(x, weights1) + bias1)
    weights2 = tf.Variable(tf.random_normal([6, 2]), name='weights2')
    bias2 = tf.Variable(tf.zeros([2]), name='bias2')
    z = tf.matmul(a, weights2) + bias2
    y_pred = tf.nn.softmax(z)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=z))
    correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    acc_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    train_op = tf.train.AdamOptimizer(0.001).minimize(cost)
