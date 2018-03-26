# -*- coding: utf-8 -*

'''
data importion
'''
import numpy as np  # for matrix calculation
import matplotlib.pyplot as plt

# load the CSV file as a numpy matrix
dataset = np.loadtxt('data.csv', delimiter=",")

# separate the data from the target attributes
X = dataset[:, 1:3] #获取x变量
y = dataset[:, 3]   # 获取结果值

m, n = np.shape(X)

# draw scatter diagram to show the raw data
f1 = plt.figure(1)
plt.title('watermelon_3a')
plt.xlabel('density')
plt.ylabel('ratio_sugar')
plt.scatter(X[y == 0, 0], X[y == 0, 1], marker='o', color='red', s=100, label='bad') # 设置坏瓜
plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='*', color='green', s=100, label='good') # 设置好瓜
plt.legend(loc='upper left')
# plt.show()

''' 
using sklearn lib for logistic regression
'''
from sklearn import metrics
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import matplotlib.pylab as pl

# generalization of test and train set


"""
作用：要是用于交叉验证函数 功能是从样本中随机的按比例选取train data和testdata

参数
---
arrays：样本数组，包含特征向量和标签

test_size：
　　float-获得多大比重的测试样本 （默认：0.25）
　　int - 获得多少个测试样本

训练样本的目的是 数学模型的参数，经过训练之后，可以认为你的模型系统确立了下来。

建立的模型有多好，和真实事件的差距大不大，既可以认为是测试样本的目的。

一般训练样本和测试样本相互独立，使用不同的数据

train_size: 同test_size

random_state:
　　int - 随机种子（种子固定，实验可复现）
   随机数种子：其实就是该组随机数的编号，在需要重复试验的时候，保证得到一组一样的随机数。比如你每次都填1，其他参数一样的情况下你得到的随机数组是一样的。但填0或不填，每次都会不一样。
   随机数的产生取决于种子，随机数和种子之间的关系遵从以下两个规则：
种子不同，产生不同的随机数；种子相同，即使实例不同也产生相同的随机数。
　　
shuffle - 是否在分割之前对数据进行洗牌（默认True）

返回
---
分割后的列表，长度=2*len(arrays), 
　　(train-test split)
"""
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.4,train_size=0.5, random_state=0)

# model training
log_model = LogisticRegression()  # using log-regression lib model
log_model.fit(X_train, y_train)  # fitting 训练

# model validation
y_pred = log_model.predict(X_test) # 预测

# summarize the fit of the model
print("confusion_matrix:")
print(metrics.confusion_matrix(y_test, y_pred)) #混淆矩阵 它是一种特定的矩阵用来呈现算法性能的可视化效果 数据越集中于对角线 代表数据越准确
print(metrics.classification_report(y_test, y_pred))

precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_pred)

# show decision boundary in plt
# X - some data in 2dimensional np.array
f2 = plt.figure(2)
h = 0.001
x0_min, x0_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
x1_min, x1_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
x0, x1 = np.meshgrid(np.arange(x0_min, x0_max, h),
                     np.arange(x1_min, x1_max, h))

# here "model" is your model's prediction (classification) function
z = log_model.predict(np.c_[x0.ravel(), x1.ravel()])

# Put the result into a color plot
z = z.reshape(x0.shape)
plt.contourf(x0, x1, z, cmap=pl.cm.Paired)

# Plot also the training pointsplt.title('watermelon_3a')
plt.title('watermelon_3a')
plt.xlabel('density')
plt.ylabel('ratio_sugar')
plt.scatter(X[y == 0, 0], X[y == 0, 1], marker='o', color='k', s=100, label='bad')
plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='o', color='g', s=100, label='good')
# plt.show()

'''
coding to implement logistic regression
'''
from sklearn import model_selection

import self_def;

# X_train, X_test, y_train, y_test
np.ones(n)
m, n = np.shape(X)
X_ex = np.c_[X, np.ones(m)]  # extend the variable matrix to [x, 1]
X_train, X_test, y_train, y_test = model_selection.train_test_split(X_ex, y, test_size=0.5, random_state=0)

# using gradDescent to get the optimal parameter beta = [w, b] in page-59
beta = self_def.gradDscent_2(X_train, y_train)

# prediction, beta mapping to the model
y_pred = self_def.predict(X_test, beta)

m_test = np.shape(X_test)[0]
# calculation of confusion_matrix and prediction accuracy
cfmat = np.zeros((2, 2))
for i in range(m_test):
    if y_pred[i] == y_test[i] == 0:
        cfmat[0, 0] += 1
    elif y_pred[i] == y_test[i] == 1:
        cfmat[1, 1] += 1
    elif y_pred[i] == 0:
        cfmat[1, 0] += 1
    elif y_pred[i] == 1:
        cfmat[0, 1] += 1

print(cfmat)