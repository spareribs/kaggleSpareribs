#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @Time    : 2018/11/18 12:12
# @Author  : Spareribs
# @File    : sklearn_train.py
# @Software: PyCharm
"""
import os
import sys

sys.path.append(os.path.abspath('.'))

import pickle
# 去除 warnings 的警告
import warnings

import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, recall_score, accuracy_score
from sklearn.model_selection import train_test_split

from sklearn_config import features_path, clfs, status_vali

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
else:
    x_train, x_vali, y_train, y_vali = x_train, x_train, y_train, y_train

"""=====================================================================================================================
2 训练分类器, clf_name选择需要的分类器
"""
clf_name = "svm_ploy"
pre_vali_dict = {}
clf = clfs[clf_name]
clf.fit(x_train, y_train)

"""=====================================================================================================================
3 模型评估 输出所有模型的分数
"""


def model_metrics(clf, x_train, x_vali, y_train, y_vali):
    print("测试模型 & 模型参数如下：\n{0}".format(clf))
    # pre_train = clf.predict(x_train)
    # print("训练集正确率: {0:.4f}".format(clf.score(x_train, y_train)))
    # print("训练集f1分数: {0:.4f}".format(f1_score(y_train, pre_train)))
    # print("训练集auc分数: {0:.4f}".format(roc_auc_score(y_train, pre_train)))
    y_train_pred = clf.predict(x_train)
    y_vali_pred = clf.predict(x_vali)
    y_train_pred_proba = clf.predict_proba(x_train)[:, 1]
    y_vali_pred_proba = clf.predict_proba(x_vali)[:, 1]

    print("=" * 20)
    # 准确性
    print("准确性: \n训练集: {0:.4f}\n测试集: {1:.4f}".format(
        accuracy_score(y_train, y_train_pred),
        accuracy_score(y_vali, y_vali_pred)
    ))
    print("-" * 20)
    # 召回率
    print("召回率: \n训练集: {0:.4f}\n测试集: {1:.4f}".format(
        recall_score(y_train, y_train_pred),
        recall_score(y_vali, y_vali_pred)
    ))
    print("-" * 20)
    # f1_score
    print("f1_score: \n训练集: {0:.4f}\n测试集: {1:.4f}".format(
        f1_score(y_train, y_train_pred),
        f1_score(y_vali, y_vali_pred)
    ))
    print("-" * 20)
    # roc_auc
    roc_auc_train = roc_auc_score(y_train, y_train_pred_proba),
    roc_auc_vali = roc_auc_score(y_vali, y_vali_pred_proba)

    print("roc_auc: \n训练集: {0:.4f}\n测试集: {1:.4f}".format(roc_auc_train[0], roc_auc_vali))
    print("-" * 20)
    # 描绘 ROC 曲线
    fpr_tr, tpr_tr, _ = roc_curve(y_train, y_train_pred_proba)
    fpr_te, tpr_te, _ = roc_curve(y_vali, y_vali_pred_proba)
    print("描绘 ROC 曲线: \n训练集: fpr_tr {0} tpr_tr {1}\n测试集: fpr_tr {2} tpr_tr {3}".format(
        len(fpr_tr), len(tpr_tr),
        len(fpr_te), len(tpr_te)
    ))
    print("-" * 20)
    # KS
    ks_train = max(abs((fpr_tr - tpr_tr))),
    ks_vali = max(abs((fpr_te - tpr_te)))
    print("KS: \n训练集: {0:.4f}\n测试集: {1:.4f}".format(
        ks_train[0],
        ks_vali
    ))
    print("=" * 20)
    rou_auc = {
        "roc_auc_train": roc_auc_train[0],
        "roc_auc_vali": roc_auc_vali,
        "ks_train": ks_train[0],
        "ks_vali": ks_vali,
        "fpr_tr": fpr_tr,
        "tpr_tr": tpr_tr,
        "fpr_te": fpr_te,
        "tpr_te": tpr_te,
    }
    return rou_auc


if status_vali:
    rou_auc = model_metrics(clf, x_train, x_vali, y_train, y_vali)
"""
4. 绘制图
https://yq.aliyun.com/articles/623375
http://bei.dreamcykj.com/2018/08/19/ROC原理介绍及利用python实现二分类和多分类的ROC曲线 (1)/
"""
color_dict = {
    "lr": "r-", 'svm': "b-", 'rf': "g-", 'xgb': "y-", 'lgb': "d-"
}
plt.plot(rou_auc.get("fpr_tr"), rou_auc.get("tpr_tr"), 'r-',
         label="Train:AUC: {:.3f} KS:{:.3f}".format(rou_auc.get("roc_auc_train"), rou_auc.get("ks_train")))
plt.plot(rou_auc.get("fpr_te"), rou_auc.get("tpr_te"), 'g-',
         label="Test:AUC: {:.3f} KS:{:.3f}".format(rou_auc.get("roc_auc_vali"), rou_auc.get("ks_vali")))
plt.plot([0, 1], [0, 1], 'd--')
plt.legend(loc='best')
plt.title("{0} ROC curse".format(clf_name))
plt.savefig("{0}_roc_auc.jpg".format(clf_name))
# plt.show()
