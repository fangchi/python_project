import numpy as np
import seaborn as sns
import  matplotlib
import matplotlib.pyplot as plt

myfont = matplotlib.font_manager.FontProperties(fname="/Library/Fonts/华文仿宋.ttf")#"/Library/Fonts/Songti.ttc")
sns.set(style="white", color_codes=True,font=myfont.get_name())

iris = sns.load_dataset("data",data_home="/Users/fangchi/PycharmProjects/python_project/marchineLearning/ch3/3.4")


iris.plot(kind="scatter", x="萼片_长度", y="萼片_宽度")
sns.pairplot(iris,hue='品种')
plt.show()

