**目录**
- <a href="#jj">`简介`</a>
- <a href="#dmsm">`代码说明`</a>
    - <a href="#dmmljg">`代码目录结构`</a>
    - <a href="#dmsyff">`代码使用方法`</a>
- <a href="#ckwd">`参考文档`</a>
    - <a href="#rwxq-rw1">`【2018.11.14 - 2018.11.15】任务1. 逻辑回归模型实践`</a>
    - <a href="#rwxq-rw2">`【2018.11.15 - 2018.11.16】任务2. 支持向量机和决策树模型实践 `</a>
    - <a href="#rwxq-rw3">`【2018.11.16 - 2018.11.18】任务3. 构建xgboost和lightgbm模型进行预测`</a>
    - <a href="#rwxq-rw4">`【2018.11.19 - 2018.11.20】任务4. 记录五个模型关于的评分表格, 画出auc和roc曲线图`</a>

<a id="jj"/>

# 简介 

12个人的小组练习任务 - 提升算法实践能力. <br>

【数据】数据是金融数据, 我们要做的是预测贷款用户是否会逾期. 表格中, status是标签: 0表示未逾期, 1表示逾期.<br>
【学习过程】: 构建模型 - 模型融合 - 模型评估 - 交叉验证 - 模型调参 - 特征工程<br>
【遵循】一次只做一件事, 先实现再优化<br>
【期望目标】掌握数据挖掘的流程, 提升合作的能力.

<a id="dmsm"/>

# 代码说明

代码目录 https://github.com/spareribs/kaggleSpareribs/blob/master/Overdue/Readme.md

<a id="dmmljg"/>

## 代码目录结构
```
Overdue
├─dl: 深度学习 TODO
├─ml: 机器学习
│  ├─code
│  │  ├─ sklearn_config.py: 模型配置文件
│  |  ├─ sklearn_gcv.py: 模型配置 网络搜索
│  |  └─ sklearn_train.py: 模型训练
│  ├─data: 数据存放的目录
│  ├─features: 
│  |  └─ base.py: 数据预处理
│  └─for beginner: TODO
└─config.py: 全局配置
```

<a id="dmsyff"/>

## 代码使用方法
1. 【必须】config.py 设置文件存放的路径
2. 【必须】先执行 features 中的 base.py 先把数据处理好 [PS:需要根据实际情况修改]
3. 【可选】再通过 code 中的 sklearn_gcv.py 搜索模型的最佳配置
4. 【必须】最后通过 code 中的 sklearn_train.py 训练模型输出结果


<a id="ckwd"/>

# 参考文档

<a id="rwxq-rw1"/>

## 任务1. 逻辑回归模型实践【2018.11.14 - 2018.11.15】
- 爖：https://github.com/LongJH/ALittleTarget/blob/master/Mission1/mission1-lr.ipynb
- Ash：https://blog.csdn.net/truffle528/article/details/84072452
- 憨宝宝：https://blog.csdn.net/qq_41205464/article/details/84111934
- 黑桃，等到的过去 共同完成：https://blog.csdn.net/lgy54321/article/details/84101357
- 排骨 https://blog.csdn.net/q370835062/article/details/84133789
- 面朝大海 https://blog.csdn.net/zhangyunpeng0922/article/details/84106715
- 大范先生，月光疾风 共同完成：https://blog.csdn.net/weixin_40671804/article/details/84111029 
- jepson：[https://github.com/JepsonWong/Algorithm_Competition/blob/master/客户预期分析/test.ipynb](https://github.com/JepsonWong/Algorithm_Competition/blob/master/%E5%AE%A2%E6%88%B7%E9%80%BE%E6%9C%9F%E5%88%86%E6%9E%90/test.ipynb)
- 李碧涵：https://github.com/libihan/Exercise-ML/blob/master/Finance.ipynb
 
<a id="rwxq-rw2"/>

## 任务2.支持向量机和决策树模型实践 【2018.11.15 - 2018.11.16】
- 爖：https://github.com/LongJH/ALittleTarget/blob/master/Mission1/mission2-svm-dt.ipynb
- 憨宝宝：https://blog.csdn.net/qq_41205464/article/details/84169197
- Ash：https://blog.csdn.net/truffle528/article/details/84168200
- 黑桃：https://blog.csdn.net/Heitao5200/article/details/84141345
- 等到的过去：https://blog.csdn.net/lgy54321/article/details/84145213
- 面朝大海：https://blog.csdn.net/zhangyunpeng0922/article/details/84136003
- 排骨：https://blog.csdn.net/q370835062/article/details/84173260
- 大范先生：https://blog.csdn.net/weixin_40671804/article/details/84144980
- 月光疾风：https://yezuolin.com/2018/11/UserLoanOverdue/
- jepson：[https://github.com/JepsonWong/Algorithm_Competition/blob/master/客户预期分析/test.ipynb](https://github.com/JepsonWong/Algorithm_Competition/blob/master/%E5%AE%A2%E6%88%B7%E9%80%BE%E6%9C%9F%E5%88%86%E6%9E%90/test.ipynb)
- 李碧涵：https://blog.csdn.net/a786150017/article/details/84138846

<a id="rwxq-rw3"/>

## 任务3.构建xgboost和lightgbm模型进行预测【2018.11.16 - 2018.11.18】
- 爖：https://github.com/LongJH/ALittleTarget/blob/master/Mission1/mission3-xgboost-lightgbm.ipynb
- 憨宝宝：https://blog.csdn.net/qq_41205464/article/details/84204927
- Ash：https://blog.csdn.net/truffle528/article/details/84200976
- 黑桃：https://blog.csdn.net/Heitao5200/article/details/84196023
- 等到的过去：https://blog.csdn.net/lgy54321/article/details/84202770
- 面朝大海：https://blog.csdn.net/zhangyunpeng0922/article/details/84193403
- 排骨：https://blog.csdn.net/q370835062/article/details/84173260
- 大范先生：https://blog.csdn.net/weixin_40671804/article/details/84186625
- 月光疾风：https://yezuolin.com/2018/11/UserLoanOverdue_XGBoost&LightGBM/
- jepson：
- 李碧涵：https://blog.csdn.net/a786150017/article/details/84138846

<a id="rwxq-rw4"/>

## 任务4.记录五个模型的评分表格，画出auc和roc曲线图【2018.11.19 - 2018.11.20】
