#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @Time    : 2018/11/18 12:12
# @Author  : Spareribs
# @File    : base.py
# @Software: PyCharm
"""

import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, r2_score, roc_auc_score
from sklearn_config import features_path, clfs, status_vali

# 去除 warnings 的警告
import warnings

warnings.filterwarnings('ignore')

"""=====================================================================================================================
1 读取数据
"""
data_fp = open(features_path, 'rb')
x_train, y_train = pickle.load(data_fp)
data_fp.close()

"""划分训练集和验证集，验证集比例为test_size"""
if status_vali:
    x_train, x_vali, y_train, y_vali = train_test_split(x_train, y_train, test_size=0.3, random_state=0)

"""=====================================================================================================================
2 训练分类器, clf_name选择需要的分类器
"""
clf_name = "lr  "
clf = clfs[clf_name]
clf.fit(x_train, y_train)

"""=====================================================================================================================
3 在验证集上评估模型
"""
if status_vali:
    print("测试模型 & 模型参数如下：\n{0}".format(clf))
    print("=" * 20)
    pre_train = clf.predict(x_train)
    print("训练集正确率: {0:.4f}".format(clf.score(x_train, y_train)))
    print("训练集f1分数: {0:.4f}".format(f1_score(y_train, pre_train)))
    print("训练集auc分数: {0:.4f}".format(roc_auc_score(y_train, pre_train)))
    print("-" * 20)
    pre_vali = clf.predict(x_vali)
    print("测试集正确率: {0:.4f}".format(clf.score(x_vali, y_vali)))
    print("测试集f1分数: {0:.4f}".format(f1_score(y_vali, pre_vali)))
    print("测试集auc分数: {0:.4f}".format(roc_auc_score(y_vali, pre_vali)))
    print("=" * 20)

