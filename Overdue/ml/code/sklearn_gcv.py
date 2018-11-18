#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @Time    : 2018/11/18 18:46
# @Author  : Spareribs
# @File    : sklearn_gcv.py
# @Software: PyCharm
"""

import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
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
clf_name = "lr"
clf = clfs[clf_name]

"""=====================================================================================================================
3 使用网络搜索获得最优的参数
"""

param_grid = {
    'C': [0.05, 0.1, 0.5, 1.5],
    'penalty': ['l1', 'l2']
}

grid = GridSearchCV(clf, param_grid, scoring='f1_micro')
grid.fit(x_train, y_train)

print("最优参数：{0}".format(grid.best_params_))
print("最好的分数{0}".format(grid.best_score_))
