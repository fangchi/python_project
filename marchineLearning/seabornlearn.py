import seaborn as sns
import  matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Seaborn其实是在matplotlib的基础上进行了更高级的API封装，从而使得作图更加容易

myfont = matplotlib.font_manager.FontProperties(fname="/Library/Fonts/华文仿宋.ttf")#"/Library/Fonts/Songti.ttc")
#set_style( )是用来设置主题的，Seaborn有五个预设好的主题： darkgrid , whitegrid , dark , white ,和 ticks  默认： darkgrid
sns.set(style="whitegrid",palette="muted", color_codes=True,font=myfont.get_name())

# eg1
# plt.plot(np.arange(10))
# plt.show()

# eg2
# iris = sns.load_dataset("data",data_home="/Users/fangchi/PycharmProjects/python_project/marchineLearning/ch3/3.4")
# fig, axes = plt.subplots(2,2)
# sns.distplot(iris['花瓣_宽度'], ax = axes[0][0], kde = True, rug = True)        # kde 密度曲线  rug 边际毛毯
# sns.kdeplot(iris['花瓣_长度'], ax = axes[0][1], shade=True) # shade  阴影
# sns.distplot(iris['萼片_长度'], ax = axes[1][0], kde = True, rug = True)
# plt.show()

#eg3
rs = np.random.RandomState(23355)
#这里看以看到，有一个23355这个数字，其实，它是伪随机数产生器的种子，也就是“the starting point for a sequence of pseudorandom number”
#对于某一个伪随机数发生器，只要该种子（seed）相同，产生的随机数序列就是相同的
# d = rs.normal(size=1000)
# f, axes = plt.subplots(2, 2, figsize=(7, 10), sharex=False)
# sns.distplot(d, kde=False, color="b", ax=axes[0, 0])
# sns.distplot(d, hist=False, rug=True, color="r", ax=axes[0, 1])
# sns.distplot(d, hist=False, color="g", kde_kws={"shade": True}, ax=axes[1, 0])
# sns.distplot(d, color="m", ax=axes[1, 1])
# plt.show()

#eg4 箱型图
# iris = sns.load_dataset("data",data_home="/Users/fangchi/PycharmProjects/python_project/marchineLearning/ch3/3.4")
# sns.boxplot(x = iris['萼片_长度'],y = iris['品种'])
# plt.show()

#eg5
# iris = sns.load_dataset("data",data_home="/Users/fangchi/PycharmProjects/python_project/marchineLearning/ch3/3.4")
# sns.jointplot("萼片_长度", "花瓣_长度", iris)
# plt.show()

#eg6
iris = sns.load_dataset("data",data_home="/Users/fangchi/PycharmProjects/python_project/marchineLearning/ch3/3.4")

plt.figure(figsize=(12,8))

sns.pointplot(iris.萼片_长度.values, iris.品种.values, alpha=0.8, color='blue')
plt.ylabel('品种', fontsize=12)
plt.xlabel('萼片_长度', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()