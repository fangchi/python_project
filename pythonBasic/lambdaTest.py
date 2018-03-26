# import pymysql
#
# fun= lambda x,y:x+y   #冒号前的x,y为自变量，冒号后x+y为具体运算。
# x=3    #这里要定义int整数，否则会默认为字符串
# y=6
# print(fun(x,y))

import copy
a=[1,2,3]
b=a
print(id(a))
print(id(b))