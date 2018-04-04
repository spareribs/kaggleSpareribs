#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @Time    : 2018/4/3 20:10
# @Author  : Spareribs
# @File    : common.py

matplotlib.pyplot使用参考文档
https://matplotlib.org/api/pyplot_summary.html


https://absentm.github.io/2017/03/18/Python-matplotlib-%E6%95%B0%E6%8D%AE%E5%8F%AF%E8%A7%86%E5%8C%96/
"""

import os
import pandas as pd  # 数据分析
import numpy as np  # 科学计算
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import sklearn.preprocessing as preprocessing
from sklearn import linear_model

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def get_data_path(filename=""):
    demo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
    data_path = os.path.join(demo_path, "data", filename)
    return data_path


def get_some_plt(data_train):
    """
    乘客各属性分布
    :return:
    """
    fig = plt.figure()
    fig.set(alpha=0.2)  # 设定图表颜色alpha参数

    plt.subplot2grid((2, 3), (0, 0))  # 在一张大图里分列几个小图
    data_train.Survived.value_counts().plot(kind='bar')  # 柱状图
    plt.title(u"获救情况 (1为获救)")  # 标题
    plt.ylabel(u"人数")

    plt.subplot2grid((2, 3), (0, 1))
    data_train.Pclass.value_counts().plot(kind="bar")
    plt.ylabel(u"人数")
    plt.title(u"乘客等级分布")

    plt.subplot2grid((2, 3), (0, 2))
    plt.scatter(data_train.Survived, data_train.Age)
    plt.ylabel(u"年龄")  # 设定纵坐标名称
    plt.grid(b=True, which='major', axis='y')
    plt.title(u"按年龄看获救分布 (1为获救)")

    plt.subplot2grid((2, 3), (1, 0), colspan=2)
    data_train.Age[data_train.Pclass == 1].plot(kind='kde')
    data_train.Age[data_train.Pclass == 2].plot(kind='kde')
    data_train.Age[data_train.Pclass == 3].plot(kind='kde')
    plt.xlabel(u"年龄")  # plots an axis lable
    plt.ylabel(u"密度")
    plt.title(u"各等级的乘客年龄分布")
    plt.legend((u'头等舱', u'2等舱', u'3等舱'), loc='best')  # sets our legend for our graph.

    plt.subplot2grid((2, 3), (1, 2))
    data_train.Embarked.value_counts().plot(kind='bar')
    plt.title(u"各登船口岸上船人数")
    plt.ylabel(u"人数")
    plt.show()


def get_pclass_plt(data_train):
    """
    看看各乘客等级的获救情况
    :return:
    """
    fig = plt.figure()
    fig.set(alpha=0.2)  # 设定图表颜色alpha参数

    Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
    Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
    df = pd.DataFrame({u'获救': Survived_1, u'未获救': Survived_0})
    df.plot(kind='bar', stacked=True)
    plt.title(u"各乘客等级的获救情况")
    plt.xlabel(u"乘客等级")
    plt.ylabel(u"人数")
    plt.show()


def get_sex_plt(data_train):
    """
    看看各性别的获救情况
    :return:
    """
    fig = plt.figure()
    fig.set(alpha=0.2)  # 设定图表颜色alpha参数

    Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
    Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()
    df = pd.DataFrame({u'男性': Survived_m, u'女性': Survived_f})
    df.plot(kind='bar', stacked=True)
    plt.title(u"按性别看获救情况")
    plt.xlabel(u"性别")
    plt.ylabel(u"人数")
    plt.show()


def get_sex_pclass_plt(data_train):
    """
    然后我们再来看看各种舱级别情况下各性别的获救情况
    :return:
    """
    fig = plt.figure()
    fig.set(alpha=0.65)  # 设置图像透明度，无所谓
    plt.title(u"根据舱等级和性别的获救情况")

    ax1 = fig.add_subplot(141)
    data_train.Survived[data_train.Sex == 'female'][data_train.Pclass != 3].value_counts().plot(kind='bar',
                                                                                                label="female highclass",
                                                                                                color='#FA2479')
    ax1.set_xticklabels([u"获救", u"未获救"], rotation=0)
    ax1.legend([u"女性/高级舱"], loc='best')

    ax2 = fig.add_subplot(142, sharey=ax1)
    data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 3].value_counts().plot(kind='bar',
                                                                                                label='female, low class',
                                                                                                color='pink')
    ax2.set_xticklabels([u"未获救", u"获救"], rotation=0)
    plt.legend([u"女性/低级舱"], loc='best')

    ax3 = fig.add_subplot(143, sharey=ax1)
    data_train.Survived[data_train.Sex == 'male'][data_train.Pclass != 3].value_counts().plot(kind='bar',
                                                                                              label='male, high class',
                                                                                              color='lightblue')
    ax3.set_xticklabels([u"未获救", u"获救"], rotation=0)
    plt.legend([u"男性/高级舱"], loc='best')

    ax4 = fig.add_subplot(144, sharey=ax1)
    data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 3].value_counts().plot(kind='bar',
                                                                                              label='male low class',
                                                                                              color='steelblue')
    ax4.set_xticklabels([u"未获救", u"获救"], rotation=0)
    plt.legend([u"男性/低级舱"], loc='best')

    plt.show()


def get_embarked_plt(data_train):
    fig = plt.figure()
    fig.set(alpha=0.2)  # 设定图表颜色alpha参数

    Survived_0 = data_train.Embarked[data_train.Survived == 0].value_counts()
    Survived_1 = data_train.Embarked[data_train.Survived == 1].value_counts()
    df = pd.DataFrame({u'获救': Survived_1, u'未获救': Survived_0})
    df.plot(kind='bar', stacked=True)
    plt.title(u"各登录港口乘客的获救情况")
    plt.xlabel(u"登录港口")
    plt.ylabel(u"人数")

    plt.show()


def get_sibsp_parch(data_train):
    g = data_train.groupby(['SibSp', 'Survived'])
    df = pd.DataFrame(g.count()['PassengerId'])
    print(df)

    g = data_train.groupby(['Parch', 'Survived'])
    df = pd.DataFrame(g.count()['PassengerId'])
    print(df)


def get_cabin_plt(data_train):
    """
    有无Cabin信息这个粗粒度上看看Survived的情况
    :param data_train:
    :return:
    """
    fig = plt.figure()
    fig.set(alpha=0.2)  # 设定图表颜色alpha参数

    Survived_cabin = data_train.Survived[pd.notnull(data_train.Cabin)].value_counts()
    Survived_nocabin = data_train.Survived[pd.isnull(data_train.Cabin)].value_counts()
    df = pd.DataFrame({u'有': Survived_cabin, u'无': Survived_nocabin}).transpose()
    df.plot(kind='bar', stacked=True)
    plt.title(u"按Cabin有无看获救情况")
    plt.xlabel(u"Cabin有无")
    plt.ylabel(u"人数")
    plt.show()


if __name__ == "__main__":
    train_path = get_data_path("train.csv")  # 获取文件路径
    data_train = pd.read_csv(train_path)
    print(u"[Info]: DataSet all\n{0}".format(data_train))  # 输出文件内容
    print(u"[Info]: DataSet info")  # 输出简单信息
    data_train.info()

    get_some_plt(data_train)  # 用图表更加直观看数据
    get_pclass_plt(data_train)  # 看看各乘客等级的获救情况
    get_sex_plt(data_train)  # 看看各性别的获救情况
    get_sex_pclass_plt(data_train)  # 看看各种舱级别情况下各性别的获救情况
    get_embarked_plt(data_train)  # 看看各登船港口的获救情况。
    get_sibsp_parch(data_train)  # 看看堂兄弟/妹，孩子/父母有几人，对是否获救的影响。
    get_cabin_plt(data_train)  # 有无Cabin信息这个粗粒度上看看Survived的情况
