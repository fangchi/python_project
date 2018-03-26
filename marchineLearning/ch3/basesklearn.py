from sklearn import datasets;

from sklearn import metrics;


iris = datasets.load_iris() # 导入数据集
X = iris.data # 获得其特征向量
y = iris.target # 获得样本label
# print(iris)
#混淆矩阵
print(metrics.confusion_matrix(y_true = [0,0,1,1,1],y_pred= [1,1,1,1,0]))


print(metrics.classification_report(y_true = [0,0,1,1,1],y_pred= [1,0,1,1,0]))