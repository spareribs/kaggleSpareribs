#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @Time    : 2018/11/18 12:12
# @Author  : Spareribs
# @File    : base.py
# @Software: PyCharm
"""

import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from config import PATH

# 去除 warnings 的警告
import warnings

warnings.filterwarnings('ignore')

# 设置显示窗口的大小
# 参考文章:
# http://sofasofa.io/forum_main_post.php?postid=1000912
# https://blog.csdn.net/saltriver/article/details/78144984
pd.set_option('display.max_row', 1000)
pd.set_option('display.max_columns', 10000)
pd.set_option('display.width', 10000)

"""====================================================================
1. 读取训练数据
"""
# 因为数据并非utf-8编码，要使用gbk编码读入，否则出错
train = pd.read_csv('{0}/data.csv'.format(PATH), encoding='gbk', dtype=object)

"""====================================================================
2. 特征工程 - 简单数据分析 目前简单分析
"""

# 2.1 确认数据占比 0表示未逾期，1表示逾期
# 0未逾期 : 1逾期
#   3561 : 1193
print(u"{0} 当前数据占比如下: ".format("*" * 20))
print(train.groupby(['status']).count())

# 2.2 简单分析数据表数据
# 需要分析每一个特征具体的作用 TODO
print(train.head().T)
print(train.describe().T)

"""====================================================================
3. 特征工程 - 数据预处理
根据2分析的结果对数据进行处理
"""

# 3.1 直接删除的数据数据
train = train.drop(["trade_no", "bank_card_no", "id_name", "Unnamed: 0", "custid", "source"], axis=1)
print(train.shape)
# 3.2 离散化处理
train.groupby(['reg_preference_for_trad']).count()
train["reg_preference_for_trad"] = train["reg_preference_for_trad"].replace("一线城市", "1")
train["reg_preference_for_trad"] = train["reg_preference_for_trad"].replace("二线城市", "2")
train["reg_preference_for_trad"] = train["reg_preference_for_trad"].replace("三线城市", "3")
train["reg_preference_for_trad"] = train["reg_preference_for_trad"].replace("其他城市", "4")
train["reg_preference_for_trad"] = train["reg_preference_for_trad"].replace("境外", "5")
# one-hot编码
# 参考文章：
# https://blog.csdn.net/cymy001/article/details/78576128
# pd.get_dummies(train["reg_preference_for_trad"], prefix="reg_preference_for_trad")
# 参考文章：https://blog.csdn.net/qq_36523839/article/details/80382924
train = pd.get_dummies(train, columns=["reg_preference_for_trad"], prefix="reg_preference_for_trad")

# 3.3 针对日期数据的处理(转换成年月日) - 后续处理，先直接删除 TODO
train = train.drop(["first_transaction_time", "latest_query_time", "loans_latest_time"], axis=1)

# 3.4 缺失值填充 - 先使用众数填充
# train.info()  # 查看数据类型
print("{0}\n     % freature".format("*" * 20))
for feature in train.columns:  # 查看缺失的数据占比
    null_count = train[feature].isnull().sum()
    print("{0:.4f} {1}".format(null_count * 100 / len(train), feature))

train = train.fillna(0)  # 使用 0 替换所有 NaN 的值
col = train.columns.tolist()[1:]


def missing(df, columns):
    """
    使用众数填充缺失值
    df[i].mode()[0] 获取众数第一个值
    """
    col = columns
    for i in col:
        df[i].fillna(df[i].mode()[0], inplace=True)
        df[i] = df[i].astype('float64')  # 将所有数据转换成 float64类型的数据


missing(train, col)

"""====================================================================
4. 归一化处理
参考文章：https://www.cnblogs.com/hudongni1/articles/5499307.html
"""
x_train = train.drop(["status"], axis=1)
y_train = train["status"]

standardScaler = StandardScaler()
scaler = standardScaler.fit(train)
train = scaler.transform(train)
print(train)

"""====================================================================
5. 将处理后的结果保存
"""
data = (x_train, y_train)
f_data = open('{0}/data_train.pkl'.format(PATH), 'wb')
pickle.dump(data, f_data)
f_data.close()
