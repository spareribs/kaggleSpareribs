#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @Time    : 2018/11/28 23:45
# @Author  : Spareribs
# @File    : base.py
# @Software: PyCharm
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, r2_score, roc_auc_score
from ml.code.sklearn_config import clfs, status_vali
from config import PATH

# 去除 warnings 的警告
import warnings

warnings.filterwarnings('ignore')

"""特征存储的路径"""
features_path = '{0}/data_train_no_standardScaler.pkl'.format(PATH)

"""=====================================================================================================================
1 读取数据
"""
data_fp = open(features_path, 'rb')
x_train, y_train = pickle.load(data_fp)
data_fp.close()
print(type(x_train))
"""=====================================================================================================================
2 IV 值挑选特征
参考：https://blog.csdn.net/ssshi0819/article/details/80426861
"""


# 计算 IV 函数
def cal_iv(x, y, n_bins=6, null_value=np.nan, ):
    # 剔除空值
    x = x[x != null_value]

    # 若 x 只有一个值，返回 0
    if len(x.unique()) == 1 or len(x) != len(y):
        return 0

    if x.dtype == np.number:
        # 数值型变量
        if x.nunique() > n_bins:
            # 若 nunique 大于箱数，进行分箱
            x = pd.qcut(x, q=n_bins, duplicates='drop')

    # 计算IV
    groups = x.groupby([x, list(y)]).size().unstack().fillna(0)
    t0, t1 = y.value_counts().index
    groups = groups / groups.sum()
    not_zero_index = (groups[t0] > 0) & (groups[t1] > 0)
    groups['iv_i'] = (groups[t0] - groups[t1]) * np.log(groups[t0] / groups[t1])
    iv = sum(groups['iv_i'])

    return iv


fea_iv = x_train.apply(lambda x: cal_iv(x, y_train), axis=0).sort_values(ascending=False)
# print(fea_iv)

# 筛选 IV > 0.1 的特征
imp_fea_iv = fea_iv[fea_iv > 0.02].index


"""=====================================================================================================================
3 随机森林挑选特征
"""

clf_name = "rf"
forest = clfs[clf_name]
forest.fit(x_train, y_train)
# print(sorted(zip(map(lambda x: round(x, 4), forest.feature_importances_), x_train.columns), reverse=True))
rf_impc = pd.Series(forest.feature_importances_, index=x_train.columns).sort_values(ascending=False)
# print(rf_impc)


# 筛选 重要性前15个特征
imp_fea_rf = rf_impc.index[:15]

"""=====================================================================================================================
4 将 IV值 和 随机森林 挑选出来的特征结合
"""
# 合并特征并筛选出有用特征
imp_fea = list(set(imp_fea_iv) | set(imp_fea_rf))
X_imp = x_train[imp_fea]
print(type(X_imp))
# print(y_train)


"""=====================================================================================================================
5 归一化处理
"""

standardScaler = StandardScaler()
scaler = standardScaler.fit(X_imp)
X_imp = scaler.transform(X_imp)
print(X_imp)

"""====================================================================
5. 将处理后的结果保存
"""
data = (X_imp, y_train)
f_data = open('{0}/data_train_iv_rf.pkl'.format(PATH), 'wb')
pickle.dump(data, f_data)
f_data.close()
