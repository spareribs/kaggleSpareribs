features 主要是数据的预处理

目前主要是简单的数据处理, 主要是模型上面的训练
1. 需要 直接删除 的数据, 这些都是唯一的id标识, 会过拟合
    - Unnamed: 0
    - 用户ID
    - trade_no：不知道是什么，可以分析下
    - bank_card_no：卡号
    - id_name：名字
    - custid: ???
    - 'source'：只有一个值xs，无意义
    
2. 需要 离散化处理 的数据
    - reg_preference_for_trad

3. 针对日期数据的处理(转换成年月日)
    - first_transaction_time
    - latest_query_time
    - loans_latest_time

4. 缺失值的填充
    - 目前只是简单的众数填充 TODO

5. 归一化处理所有数据
    - 目前直接使用StandardScaler方法处理, 没搞明白 TODO