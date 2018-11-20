#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @Time    : 2018/11/18 14:36
# @Author  : Spareribs
# @File    : sklearn_config.py
# @Software: PyCharm
"""

# 设置不启用gpu
import os


import lightgbm as lgb
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from config import PATH

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

"""是否开启验证集模式"""
status_vali = True

"""特征存储的路径"""
features_path = '{0}/data_train.pkl'.format(PATH)

"""修改base_clf改变集成学习的基分类器"""

base_clf = LinearSVC()

clfs = {
    'lr': LogisticRegression(penalty='l1', C=0.05),
    'svm': LinearSVC(C=0.5, penalty='l2', dual=True),
    'svm_linear': SVC(kernel='linear', probability=True),
    'svm_ploy': SVC(kernel='poly', probability=True),
    'bagging': BaggingClassifier(base_estimator=base_clf, n_estimators=60, max_samples=1.0, max_features=1.0,
                                 random_state=1, n_jobs=1, verbose=1),
    'rf': RandomForestClassifier(n_estimators=10, criterion='gini'),
    'adaboost': AdaBoostClassifier(base_estimator=base_clf, n_estimators=50, algorithm='SAMME'),
    'gbdt': GradientBoostingClassifier(),
    # 'xgb': xgb.XGBClassifier(max_depth=6, learning_rate=0.1, n_estimators=100, silent=True, objective='multi:softmax',
    #                          nthread=1, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1,
    #                          colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, seed=0,
    #                          missing=None),
    'xgb': xgb.XGBClassifier(),
    'lgb': lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=31, max_depth=-1, learning_rate=0.1, n_estimators=250,
                              max_bin=255, subsample_for_bin=200000, objective=None, min_split_gain=0.0,
                              min_child_weight=0.001,
                              min_child_samples=20, subsample=1.0, subsample_freq=1, colsample_bytree=1.0,
                              reg_alpha=0.0,
                              reg_lambda=0.5, random_state=None, n_jobs=-1, silent=True)
    # 'lgb': lgb.LGBMClassifier()
}
