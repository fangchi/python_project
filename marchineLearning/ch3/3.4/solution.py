import numpy as np
import seaborn as sns
import  matplotlib
import matplotlib.pyplot as plt
# https://blog.csdn.net/Snoopy_Yuan/article/details/64131129
# myfont = matplotlib.font_manager.FontProperties(fname="/Library/Fonts/华文仿宋.ttf")#"/Library/Fonts/Songti.ttc")
# sns.set(style="white", color_codes=True,font=myfont.get_name())
#
# iris = sns.load_dataset("data",data_home="/Users/fangchi/PycharmProjects/python_project/marchineLearning/ch3/3.4")
#
#
# # iris.plot(kind="scatter", x="萼片_长度", y="萼片_宽度")
# sns.pairplot(iris,hue='品种')
# plt.show()

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import cross_val_predict
iris = sns.load_dataset("data",data_home="/Users/fangchi/PycharmProjects/python_project/marchineLearning/ch3/3.4")
X = iris.values[50:150,0:4]
y = iris.values[50:150,4]
# log-regression lib model
log_model = LogisticRegression()

# 10-folds CV  十折交叉验证，英文名叫做10-fold cross-validation，用来测试算法准确性
# 将数据集分成十份，轮流将其中9份作为训练数据，1份作为测试数据，进行试验。
y_pred = cross_val_predict(log_model, X, y, cv=10)
print("十折交叉验证:",metrics.accuracy_score(y, y_pred))

# LOO CV 留一叉验证，英文名叫做10-fold cross-validation，用来测试算法准确性
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
accuracy = 0;
for train, test in loo.split(X):
    log_model.fit(X[train], y[train])  # fitting
    y_p = log_model.predict(X[test])
    if y_p == y[test] : accuracy += 1
print(accuracy / np.shape(X)[0])

