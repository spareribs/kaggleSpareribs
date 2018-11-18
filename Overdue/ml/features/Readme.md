features 主要是数据的预处理

目前主要是简单的数据处理, 主要是模型上面的训练
1. 需要 直接删除 的数据, 这些都是唯一的id标识, 会过拟合
    - bank_card_no：只有一个值 '卡号1' , 无区分度
    - source：只有一个值 'xs' , 无区分度
    - 'Unnamed: 0': 与预测值无关 
    - custid: 与预测值无关 
    - id_name：与预测值无关 
    - trade_no：与预测值无关 
    
2. 【类别特征】需要 离散化处理 的数据
    - reg_preference_for_trad
    - regional_mobility
    - student_feature
    - is_high_user

3. 【日期】针对日期数据的处理(转换成年月日)
    - first_transaction_time
    - latest_query_time
    - loans_latest_time

4. 【删除部分特征】：统计各个列标准差，将标准差小于0.1的特征剔除
    
5. 【缺失值】缺失值的填充
    - 目前只是简单的众数填充 TODO
    - 缺失的数据作为一种新特征，衡量数据的完整度

6. 归一化处理所有数据
    - 目前直接使用StandardScaler方法处理, 没搞明白 TODO