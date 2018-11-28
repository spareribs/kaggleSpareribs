# coding: utf-8

import datetime
import time

import pickle
import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import StratifiedKFold
from ml.code.sklearn_config import features_path, clfs, status_vali
from mlxtend.classifier import StackingCVClassifier
import warnings

warnings.filterwarnings('ignore')

"""=====================================================================================================================
1 读取数据
"""
data_fp = open(features_path, 'rb')
x_train, y_train = pickle.load(data_fp)
data_fp.close()

# if status_vali:
#     x_train, x_vali, y_train, y_vali = train_test_split(x_train, y_train, test_size=0.3, random_state=0)

# x_train = x_train.reset_index(drop=True)
# x_vali = x_vali.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
# y_vali = y_vali.reset_index(drop=True)
"""=====================================================================================================================
2 模型融合；
学习参考:https://blog.csdn.net/LAW_130625/article/details/78573736
"""

lr_clf = clfs["lr"]  # meta_classifier
svm_clf = clfs["svm_ploy"]
rf_clf = clfs["rf"]
xgb_clf = clfs["xgb"]
lgb_clf = clfs["lgb"]

sclf = StackingCVClassifier(classifiers=[lr_clf, svm_clf, rf_clf, xgb_clf, lgb_clf],
                            meta_classifier=lr_clf, use_probas=True, verbose=3)

sclf.fit(x_train, y_train)

print("测试模型 & 模型参数如下：\n{0}".format(sclf))
print("=" * 20)
pre_train = sclf.predict(x_train)
print("训练集正确率: {0:.4f}".format(sclf.score(x_train, y_train)))
print("训练集f1分数: {0:.4f}".format(f1_score(y_train, pre_train)))
print("训练集auc分数: {0:.4f}".format(roc_auc_score(y_train, pre_train)))
