import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

a = np.array([4, 5, 6])
print("a的类型", type(a))  # a的类型 <class 'numpy.ndarray'>
print("a的形状", a.shape)
print("a的第一个元素为", a[0])
b = np.array([[4, 5, 6], [1, 2, 3]])
print("b的形状", b.shape)
print(b[0, 0], b[0, 1], b[1, 1])
c = np.zeros((3, 3), dtype=np.int64)  # 全0矩阵
print(c)
d = np.ones((4, 5), dtype=np.int64)
print(d)
print(type(d))  # <class 'numpy.ndarray'>
f = np.arange(12)
print(f)
print(f.shape)
f = f.reshape((3, 4))
print(f)
print(f.shape)
print(f[1, :])
print(f[:, 2:])  # 输出最后两列
print(f[2, -1])  # 用-1代表最后一个元素

num_observations = 100
x = np.linspace(-3, 3, num_observations)  # 用于生成-3到3的100个数
# print(x)
y = np.sin(x) + np.random.uniform(-0.5, 0.5, num_observations)
'''# 绘制散点图
plt.scatter(x, y, label='Data Points', color='r', marker='o')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter Plot of y vs. x')
plt.grid(True)
plt.legend()
plt.show()'''
# 指定中文字体文件路径
font_path = r"c:\windows\fonts\SIMLI.TTF"  # 替换为你的字体文件路径

# 加载字体
fontprop = fm.FontProperties(fname=font_path, size=12)
# 转换为 PyTorch 张量
x = torch.FloatTensor(x)
y = torch.FloatTensor(y)


# 定义线性回归模型
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)  # 输入维度为1，输出维度为1

    def forward(self, x):
        return self.linear(x)


# 初始化模型和优化器
model = LinearRegression()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(x.unsqueeze(1))
    loss = criterion(outputs, y.unsqueeze(1))
    loss.backward()
    optimizer.step()

# 输出参数 w 和 b 以及损失
w = model.linear.weight.item()
b = model.linear.bias.item()
print("参数 w:", w)
print("参数 b:", b)
print("损失:", loss.item())

# 生成预测数据
x_pred = np.linspace(-3, 3, 100)
y_pred = w * x_pred + b

# 绘制预测回归曲线和训练数据散点图
plt.scatter(x.numpy(), y.numpy(), label='训练数据散点', color='b', marker='o')
plt.plot(x_pred, y_pred, label='预测回归曲线', color='r',)
plt.xlabel('x',fontproperties=fontprop)
plt.ylabel('y',fontproperties=fontprop)
plt.title('线性回归模型',fontproperties=fontprop)
plt.grid(True)
# 设置图例的中文字体
plt.legend(prop=fontprop)
plt.show()


