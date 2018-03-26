import numpy as np

s = np.random.rand()  # 生成生成[0,1)之间随机浮点数

print(s)

text = 'This is my first test.\nThis is the second line.\nThis the third line'
my_file = open('data\my file.txt', 'w')  # 用法: open('文件名','形式'), 其中形式有'w':write;'r':read.
my_file.write(text)  # 该语句会写入先前定义好的 text
my_file.close()

append_text = '\nThis is appended file.'  # 为这行文字提前空行 "\n"
my_file = open('data\my file.txt', 'a')  # 'a'=append 以增加内容的形式打开
my_file.write(append_text)
my_file.close()

file = open('data\my file.txt', 'r')
content = file.readlines()  # python_list 形式
for co in content:
    print(co)

a = [4,1,2,3,4,1,1,-1]
a.sort() # 默认从小到大排序
print(a)
# [-1, 1, 1, 1, 2, 3, 4, 4]

a.sort(reverse=True) # 从大到小排序
print(a)

import time
print(time.localtime())  #这样就可以print 当地时间了

from pythonBasic import car

my_car = car.Car()
print("I'm a car!")
try:
    file=open('eeee.txt','r')  #会报错的代码
except Exception as e:  # 将报错存储在 e 中
    print(e)

