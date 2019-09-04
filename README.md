# ARLMR
全称：Auto Representation Learning and Matching for Recommendation system

来源：出自项目，要写论文，保留权利到2020年6月。（也没什么人看，主要写给自己）

大体结构：
ARLMR分两部分实验——RLMR使用普通的神经网络学习User和Item的表示并匹配得分;AC_Rec使用Actor Critic的深度强化学习网络结构实现推荐系统。
最终我将把两部分融合起来，变成最终的ARLMR。

数据集：ml-100k和ml-1m

设置原则：尽量贴近真实情况。

## RLMR
数据处理方式选择：
1、根据评分记录随机分配;
2、根据固定时间点划分;
3、对每个用户按时间采取一定比例。

任务选择：
1、回归任务：预测评分1-5;（ACCM）
2、CTR预估1：预测看或不看，有评分的为1，随机采取无评分的为0;（NAIS）
3、CTR预估2：预测喜欢不喜欢，评分4-5的为1， 其他为0。（DIN）

模型细节考虑：
1、历史记录attention得到用户偏好（要不要过滤久远的记录？历史记录过多会不会对attention造成影响？）;
2、用户偏好和用户个人信息attention得到用户表示;（融合时采用加权？Y型网络？concate？交叉+concate？）
3、用户表示和目标item表示得到评分/ctr预估值。

训练方式选择：
1、对于回归任务：采用MSE误差，梯度下降优化;
2、对于CTR预估任务：复对数似然 or pair learning。（微博）

评估方式选择：
1、回归任务：毫不犹豫RMSE。
2、CTR预估任务：论文中大多用HR和NDCG，工业中常用AUC，用precision， recall的少。

## AC_Rec
数据处理方式：
强化学习和时间息息相关，因此不用随机分配。
1、时间点划分
2、同用户按时间比例划分

强化学习设置：
用户状态：用户+历史记录。
动作：用户表示向量，用来与目标产品匹配得到预测结果，用于排序。排序结果取topk对用户进行推荐。
奖励：目前是评分4，5为1;3为0;1,2为-1。（此处应尝试一下与RLMR任务相对应。）

评估方式：
目前每个epoch统计奖励总和，采用长期平均奖励。
基于每次推荐的平均topk列表的用户反馈给出评价。

## merge
（待续）

