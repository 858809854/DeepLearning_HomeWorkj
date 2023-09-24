import numpy as np

array = np.array([[1, 2, 3],
                  [2, 3, 4]], dtype=np.float64)  # 没有逗号来分隔
print(array)
print("number of dim", array.ndim)  # 纬度
print("shape", array.shape)  # 形状
print("size", array.size)  # 元素个数
print(array.dtype)
# np.zeros()
array1 = np.zeros((3, 4), dtype=np.int64)  # 定义全部为0的矩阵
print(array1)
# np.ones()
array2 = np.ones((3, 4), dtype=np.int64)  # 定义全部为1的矩阵
print(array2)
# np.empty()
array6 = np.empty((3, 4), dtype=np.int64)  # 定义一个接近于0的矩阵
print(array6)
# np.arange()  用于生成数据
array3 = np.arange(10, 20, 2)  # 生成从10到20（不包括20），步长为2
print(array3)
array4 = np.arange(0, 12).reshape((2, 6))  # 生成从0到12（不包括12），步长为1
print(array4)
# np.linspace()  用于生成线段
array5 = np.linspace(1, 10, 6).reshape((2, 3))  # 生成1到10的6个数，可以重新指定形状
print(array5)

# 加法和减法运算
array6 = np.arange(0, 12, 1).reshape((3, 4))
array7 = np.ones((3, 4), dtype=np.int64)
print(array6)
print(array7)
array8 = array6 - array7
print(array8)
# 三角函数运算  平方运算  加减运算
array9 = np.arange(0, 12, 1).reshape((3, 4))
print(array9)
print(array9 ** 2)
print(array9 - 2)  # 可以直接加减
print(np.sin(array9))  # 使用三角函数
print(array9 == 3)  # 可以进行判断布尔值
# 矩阵的乘法  分为点乘和叉乘
ar1 = np.array([[1, 2], [3, 4]], dtype=np.int64)
ar2 = np.array([[2, 3], [4, 5]], dtype=np.int64)
#   叉乘   对应元素逐个相乘
print(ar1 * ar2)
#  点乘    做矩阵的运算np.dot()函数
print(np.dot(ar1, ar2))
print(ar1.dot(ar2))  # 不同的表达方式
#  随机生成
ar3 = np.random.random((3, 4))  # 随机生成0到1之间 三行四列的数组
print(ar3)
print(ar3.max())  # 计算整个矩阵的最大最小值
print(ar3.min())
print(ar3.sum())
print(np.average(ar3))
'print(ar3.average())'  # 这种写法是错误的
print(ar3.mean())
print(np.mean(ar3))
print(np.median(ar3))
'print(ar3.median())'  # 这种写法是错误的
#  计算某一列的最大最小值
print(np.max(ar3, axis=0))
#  计算某一行的最大最小值
print(np.min(ar3, axis=1))
print(ar3.min(axis=1))  # 这种做法也可以
#  计算最大最小值的索引
print(ar3)
print(ar3.argmin())
print(np.argmin(ar3))
print(np.argmax(ar3))
#  按序号累加
ar4 = np.ones((3, 4), dtype=np.int64).cumsum()
print(ar4)
#  后-前
ar5 = np.arange(0, 12, 1).reshape((3, 4))
print(ar5)
print(np.diff(ar5))
'print(ar5.diff())'  # 这种写法是错误的
#  输出非0元素所在位置
print(np.nonzero(ar5))  # 是分别输出行数和列数
print(ar5[0])
#  对数组进行排序
ar6 = np.arange(13, 1, -1).reshape(3, 4)
print(ar6)
print(np.sort(ar6))  # 从小到大逐行进行
# 转置 np.transpose
print(ar6)
print(ar6.T)
print(np.transpose(ar6))
#  补 全  所有小于5的都等于5，所有大于9的都等于9
print(ar6)
print(np.clip(ar6, 5, 9))
print(np.sum(ar6, axis=0))  # 可选的参数 用于对于行计算或者对于列计算
# 索引  从0开始
A = np.arange(3, 15).reshape((3, 4))
print(A)
print(A[2][1])
print(A[2, 1])
print(A[0, :])  # 这里用:代替所有的值
print(A[:, 0])
'print(A[:][0])'  # 这个是有问题的
for item in A:  # 遍历每一行
    print(item)
for column in A.T:  # 遍历每一列
    print(column)
print(A.flatten())  # 平铺并返回一个列表
for e in A.flat:  # 遍历每一项
    print(e)

# 合并
B = np.arange(1, 13, 1).reshape((3, 4))
# 分割
A = np.arange(12).reshape((3, 4))
print(A)
print(np.split(A, 2, axis=1))
# 不等量分割
print(np.array_split(A, 3, axis=1))
# 直接使用横向和纵向分割函数
print(np.vsplit(A, 3))  # 纵向
print(np.hsplit(A, 2))  # 横向

# 赋值
B = np.arange(12)
C = np.array([[0,1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]])
print(B)
print(B.shape)
print(C)
print(C.shape)
D=B
E=B.copy()  # deep copy
print(D is B)
print(D)
print(E is B)
print(E)
B[0]=100
print(B)
print(D)
print(E)  # 不会改变

