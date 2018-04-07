import numpy as np
dataset = np.loadtxt('/Users/fangchi/PycharmProjects/python_project/marchineLearning/ch3/3.3/data.csv', delimiter=",")

X = dataset[:, 1:3]
y = dataset[:, 3]

z1 = X[y == 0, 0]
z2 = X[y == 0, 1]
z3 = X[y == 1, 0]
z4 = X[y == 1, 1]
# print(y==0)



np.array([1,2,3]) # 创建一维数组
np.asarray([1,2,3])
# np.array([1,2,3], [4,5,6]) # 创建多维数组

np.zeros((3, 2)) # 3行2列 全0矩阵
np.ones((3, 2)) #全1矩阵
np.full((3, 2), 5) # 3行2列全部填充5


arr1 = np.array([[1,2,3], [4,5,6]])
arr2 = np.array([[6,5], [4,3], [2,1]])

# 查看arr维度
print(arr1.shape) # (2, 3)

#切片
np.array([1,2,3,4,5,6])[:3]  #array([1,2,3])
arr1[0:2,0:2] # 二维切片 [[1 2] [4 5]]

#乘法
np.array([1,2,3]) * np.array([2,3,4]) # 对应元素相乘 array([2,6,  12])
print(np.array([[1,2],[3,4]]).dot(np.array([[1,2],[3,4]]))) # 矩阵乘法

#矩阵求和
np.sum(arr1)  # 所有元素之和 21
np.sum(arr1, axis=0) #列求和 array([5, 7, 9])
np.sum(arr1, axis=1) # 行求和 array([ 6, 15])

# 最大最小

arr = np.array([[1,2], [3,4], [5,6]])#[[1,2], [3,4], [5,6]])

#布尔型数组访问方式
print((arr>2))

print(arr[arr>2]) # [3 4 5 6]



#修改形状
# print("修改形状")
# print(arr.reshape(2,3)) #[[1 2 3]  [4 5 6]]
# print("转置") # 转置
# print(arr.T) # 转置 @[[1 3 5] [2 4 6]]
# print("摊平") # 摊平
# print(arr.flatten()) # 摊平 array([1, 2, 3, 4, 5, 6]np.arange(-2, -8, -2))


print(np.arange(1,4, 1))
print(np.arange(-2, -8, -4))
print(np.meshgrid([5,6,7],[3,45]))
"""
print(np.meshgrid([5,6,7],[3,45]))
[array([[5, 6, 7],
       [5, 6, 7]]), array([[ 3,  3,  3],
       [45, 45, 45]])]
"""
# x3,x4 = np.meshgrid([5,6,7],[3,45])
# print(x3.ravel())
# print(x4.ravel())


c = np.array([[1,1],[1,2],[1,3],[1,4]])
print(c.shape )


print(np.random.uniform(1, 10, (5, 5)))