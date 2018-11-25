# coding: utf-8

import datetime
import time

import pickle
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from ml.code.sklearn_config import features_path, clfs, status_vali

import warnings

warnings.filterwarnings('ignore')

"""=====================================================================================================================
1 读取数据
"""
data_fp = open(features_path, 'rb')
x_train, y_train = pickle.load(data_fp)
data_fp.close()

"""=====================================================================================================================
2 进行K次训练；用K个模型分别对测试集进行预测，并得到K个结果，再进行结果的融合
学习参考:https://blog.csdn.net/wqh_jingsong/article/details/77896449
"""
print(u"[Info]: K次训练开始...")
preds = []
i = 0
my_splits = 5
skf = StratifiedKFold(n_splits=my_splits, random_state=1)
score_sum = 0
for train_idx, vali_idx in skf.split(x_train, y_train):
    i = i+1
    t_start = time.time()
    """获取训练集和验证集"""
    f_train_x = x_train[train_idx]
    f_train_y = y_train[train_idx]
    f_vali_x = x_train[vali_idx]
    f_vali_y = y_train[vali_idx]

    """训练分类器"""
    clf_name = "lgb"
    clf = clfs[clf_name]
    clf.fit(f_train_x, f_train_y)

    """=====================================================================================================================
    3 在验证集上评估模型
    """
    if status_vali:
        print("=" * 20)
        pre_vali = clf.predict(f_vali_x)
        score_vali = f1_score(y_true=f_vali_y, y_pred=pre_vali)
        print("第{}折， 验证集分数：{}".format(i, score_vali))
        score_sum += score_vali
        score_mean = score_sum / i
        print("第{}折后， 验证集分平均分数：{}".format(i, score_mean))
        t_end = time.time()
        # print("耗时:{}min".format((t_end - t_start) / 60))
        # print(u"[Info]: 当前时间是: {0}".format(datetime.datetime.now().strftime('%Y%m%d %H:%M:%S')))
