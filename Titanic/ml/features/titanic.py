#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @Time    : 2018/3/19 20:54
# @Author  : Spareribs
# @File    : titanic1.py
# @Software: PyCharm
# @Url     : https://www.kaggle.com/c/titanic
"""

import os
import pandas as pd  # 数据分析
import numpy as np  # 科学计算
from sklearn.ensemble import RandomForestRegressor
import sklearn.preprocessing as preprocessing
from sklearn import linear_model
from analysis import get_data_path
from base import logistic


def set_missing_ages(df):
    """
    使用 RandomForestClassifier 填补缺失的年龄属性
    :param df:
    :return:
    """
    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]

    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()

    # y即目标年龄
    y = known_age[:, 0]

    # X即特征属性值
    X = known_age[:, 1:]

    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)

    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1::])

    # 用得到的预测结果填补原缺失数据
    df.loc[(df.Age.isnull()), 'Age'] = predictedAges

    return df, rfr


def set_cabin_type(df):
    df.loc[(df.Cabin.notnull()), 'Cabin'] = "Yes"
    df.loc[(df.Cabin.isnull()), 'Cabin'] = "No"
    return df


def main():
    ####################
    # 1. 数据获取
    ####################
    train_path = get_data_path("train.csv")  # 获取文件路径
    data_train = pd.read_csv(train_path)

    ####################
    # 2. 数据预处理
    ####################
    # 2.1 [填补缺失值] 使用 RandomForestClassifier 填补缺失的年龄属性
    data_train, rfr = set_missing_ages(data_train)
    data_train = set_cabin_type(data_train)

    # 2.2 [类目型的特征因子化] 预处理Cabin，Embarked，Sex，Pclass的数据
    dummies_cabin = pd.get_dummies(data_train['Cabin'], prefix='Cabin')
    dummies_embarked = pd.get_dummies(data_train['Embarked'], prefix='Embarked')
    dummies_sex = pd.get_dummies(data_train['Sex'], prefix='Sex')
    dummies_pclass = pd.get_dummies(data_train['Pclass'], prefix='Pclass')
    df = pd.concat([data_train, dummies_cabin, dummies_embarked, dummies_sex, dummies_pclass], axis=1)
    df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
    # print(df)


    # 2.3 [变化幅度较大的特征化到[-1,1]之内] 预处理Age，Fare的数据
    scaler = preprocessing.StandardScaler()
    df['Age_scaled'] = scaler.fit_transform(df['Age'].reshape(-1, 1))
    df['Fare_scaled'] = scaler.fit_transform(df['Fare'].reshape(-1, 1))
    df.drop(['Age', 'Fare'], axis=1, inplace=True)
    # print(df)

    ####################
    # 3. 数据分析
    ####################
    # 用正则取出我们要的属性值
    train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    train_np = train_df.as_matrix()
    y = train_np[:, 0]  # y即Survival结果
    X = train_np[:, 1:]  # X即特征属性值
    clf = logistic(X, y)
    # print(clf)

    ####################
    # 4. 数据测试
    ####################
    test_path = get_data_path("test.csv")
    data_test = pd.read_csv(test_path)
    data_test.loc[(data_test.Fare.isnull()), 'Fare'] = 0
    # 接着我们对test_data做和train_data中一致的特征变换
    # 首先用同样的RandomForestRegressor模型填上丢失的年龄
    tmp_df = data_test[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    null_age = tmp_df[data_test.Age.isnull()].as_matrix()
    # 根据特征属性X预测年龄并补上
    X = null_age[:, 1:]
    predictedAges = rfr.predict(X)
    data_test.loc[(data_test.Age.isnull()), 'Age'] = predictedAges

    data_test = set_cabin_type(data_test)
    dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix='Cabin')
    dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix='Embarked')
    dummies_Sex = pd.get_dummies(data_test['Sex'], prefix='Sex')
    dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix='Pclass')

    df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
    df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
    df_test['Age_scaled'] = scaler.fit_transform(df_test['Age'].reshape(-1, 1))
    df_test['Fare_scaled'] = scaler.fit_transform(df_test['Fare'].reshape(-1, 1))
    print(df_test)

    ####################
    # 5. 数据保存
    ####################
    logistic_regression_predictions_path = get_data_path("logistic_regression_predictions.csv")
    test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    predictions = clf.predict(test)
    result = pd.DataFrame({
        'PassengerId': data_test['PassengerId'].as_matrix(),
        'Survived': predictions.astype(np.int32)
    })
    result.to_csv(logistic_regression_predictions_path, index=False)


if __name__ == "__main__":
    main()
