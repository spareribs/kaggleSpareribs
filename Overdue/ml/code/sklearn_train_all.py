#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @Time    : 2018/11/18 12:12
# @Author  : Spareribs
# @File    : sklearn_train.py
# @Software: PyCharm
"""

import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, r2_score, roc_auc_score, roc_curve
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
clf_names = ["lr", 'svm', 'rf', 'xgb', 'lgb']
pre_vali_dict = {}
for clf_name in clf_names:
    clf = clfs[clf_name]
    clf.fit(x_train, y_train)

    """=====================================================================================================================
    3 输出所有模型的分数
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
        pre_vali_dict[clf_name] = pre_vali

"""
4. 绘制图
https://yq.aliyun.com/articles/623375

"""
color_dict = {
    "lr": "r-", 'svm': "b-", 'rf': "g-", 'xgb': "y-", 'lgb': "d-"
}
if status_vali:
    for pre_vali in pre_vali_dict:
        fpr, tpr, thresholds = roc_curve(y_vali, pre_vali_dict[pre_vali])
        plt.plot(fpr, tpr, color_dict[pre_vali], label=pre_vali)

    plt.legend()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()
