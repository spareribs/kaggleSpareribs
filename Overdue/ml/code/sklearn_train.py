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
from sklearn.metrics import f1_score, r2_score
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
clf_name = "svm"
clf = clfs[clf_name]
clf.fit(x_train, y_train)

"""=====================================================================================================================
3 在验证集上评估模型
"""
if status_vali:
    pre_vali = clf.predict(x_vali)
    model_vali = clf.score(x_vali, y_vali)
    f1_score_vali = f1_score(y_vali, pre_vali)
    r2_score_vali = r2_score(y_vali, pre_vali)
    print("测试模型：{}".format(clf_name))
    print("模型参数如下：{0}".format(clf))
    print("验证集正确率: {}".format(model_vali))
    print("验证集f1分数: {}".format(f1_score_vali))
    print("验证集r2分数: {}".format(r2_score_vali))
