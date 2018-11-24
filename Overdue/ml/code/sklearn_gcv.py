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

"""=====================================================================================================================
2 逻辑回归 - 使用网络搜索获得最优的参数
"""
clf_name = "lr"
clf = clfs[clf_name]

param_grid = {
    'C': [0.05, 0.1, 0.5, 1.5],
    'penalty': ['l1', 'l2']
}

grid = GridSearchCV(clf, param_grid, scoring='f1_micro')
grid.fit(x_train, y_train)

print("最优参数：{0}".format(grid.best_params_))
print("最好的分数{0}".format(grid.best_score_))

"""=====================================================================================================================
3 SVM - 使用网络搜索获得最优的参数
"""
clf_name = "svm"
clf = clfs[clf_name]

param_grid = {
    'C': [0.05, 0.1, 0.5, 1.5],
    'penalty': ['l2'],
    'dual': [True]
}

grid = GridSearchCV(clf, param_grid, scoring='f1_micro')
grid.fit(x_train, y_train)

print("最优参数：{0}".format(grid.best_params_))
print("最好的分数{0}".format(grid.best_score_))

"""=====================================================================================================================
4 随机森林 - 使用网络搜索获得最优的参数
"""
clf_name = "rf"
clf = clfs[clf_name]

param_grid = {
    'criterion': ['gini'],
    'n_estimators': range(10, 71, 10)
}

grid = GridSearchCV(clf, param_grid, scoring='f1_micro')
grid.fit(x_train, y_train)

print("最优参数：{0}".format(grid.best_params_))
print("最好的分数{0}".format(grid.best_score_))


"""=====================================================================================================================
5 xgboost - 使用网络搜索获得最优的参数
"""
clf_name = "xgb"
clf = clfs[clf_name]

param_grid = {
    'max_depth': [3],
    'learning_rate': [0.1],
    'n_estimators': [50, 100, 500]
}

grid = GridSearchCV(clf, param_grid, scoring='f1_micro')
grid.fit(x_train, y_train)

print("最优参数：{0}".format(grid.best_params_))
print("最好的分数{0}".format(grid.best_score_))

"""=====================================================================================================================
6 lightgbm - 使用网络搜索获得最优的参数
"""
clf_name = "lgb"
clf = clfs[clf_name]

param_grid = {
    'boosting_type': ['gbdt'],
    'n_estimators': [100, 200, 250]
}

grid = GridSearchCV(clf, param_grid, scoring='f1_micro')
grid.fit(x_train, y_train)

print("最优参数：{0}".format(grid.best_params_))
print("最好的分数{0}".format(grid.best_score_))
