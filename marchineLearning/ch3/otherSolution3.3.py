# -*- 习题3.3 解答 -*-
# 对率回归分类
import numpy as np
from numpy import linalg
import pandas as pd

# 读取数据集
# inputfile = 'data.csv'
# data_original = pd.read_excel(inputfile)
# # 数据的初步转化与操作--属性x变量2行17列数组，并添加一组1作为吸入的偏置x^=（x;1）
# x = np.array(
#     [list(data_original[u'密度']), list(data_original[u'含糖率']), [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
# y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
print("begin")

dataset = np.loadtxt('data.csv', delimiter=",")
X = dataset[:, 1:3]
hang, lie = np.shape(X);
X_ex = np.c_[X, np.ones(hang)]
x = X_ex.T
m, n = np.shape(x);
y = dataset[:, 3]

# 定义初始参数
beta = np.array([[0], [0], [1]])  # β列向量  其他参数[[0], [0], [0]]  [[0], [0], [1.1]]
old_l = 0  # 3.27式l值的记录，这是上一次迭代的l值
n = 0

while 1:
    beta_T_x = np.dot((beta.T)[0], x)  # 对β进行转置取第一行（因为β转置后是array([[0, 0, 1]]，取第一行得到array([0, 0, 1])
    # ，再与x相乘（dot）,beta_T_x表示β转置乘以x)
    cur_l = 0  # 当前的l值
    for i in range(hang):
        cur_l = cur_l + (-y[i] * beta_T_x[i] + np.log(1 + np.exp(beta_T_x[i])))  # 计算当前3.27式的l值，这是目标函数，希望他越小越好  beta_T_x[i] = WT(x)
    # 迭代终止条件
    if np.abs(cur_l - old_l) <= 0.000001:  # 精度，二者差在0.000001以内就认为可以了，说明l已经很收敛了
        print("确实收敛")
        break  # 满足条件直接跳出循环

    # 牛顿迭代法更新β
    # 求关于β的一阶导数和二阶导数

    n = n + 1
    old_l = cur_l
    dbeta = 0
    d2beta = 0
    for i in range(hang):
        dbeta = dbeta - np.dot(np.array([x[:, i]]).T,
                               (y[i] - (np.exp(beta_T_x[i]) / (1 + np.exp(beta_T_x[i])))))  # 一阶导数 3.30
        d2beta = d2beta + np.dot(np.array([x[:, i]]).T, np.array([x[:, i]]).T.T) * (
                np.exp(beta_T_x[i]) / (1 + np.exp(beta_T_x[i]))) * (
                         1 - (np.exp(beta_T_x[i]) / (1 + np.exp(beta_T_x[i])))) # 二阶导数 3.31
    beta = beta - np.dot(linalg.inv(d2beta), dbeta)  #3.29函数
    print("当前迭代", n)
    print("当前beta", beta)
print('模型参数是：', beta)
print('迭代次数：', n)

