import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 生成数据
num_observations = 100
x = np.linspace(-3, 3, num_observations)
y = np.sin(x) + np.random.uniform(-0.5, 0.5, num_observations)
# y = 2 * x**3 - 3 * x**2 + 4 * x + np.random.normal(0, 1, num_observations)
# 转换为 PyTorch 张量
x = torch.FloatTensor(x)
y = torch.FloatTensor(y)


# 定义多元线性回归模型
class PolynomialRegression(nn.Module):
    def __init__(self):
        super(PolynomialRegression, self).__init__()
        self.linear1 = nn.Linear(1, 1)
        self.linear2 = nn.Linear(1, 1)
        self.linear3 = nn.Linear(1, 1)

    def forward(self, x):
        x1 = x
        x2 = x ** 2
        x3 = x ** 3
        out1 = self.linear1(x1)
        out2 = self.linear2(x2)
        out3 = self.linear3(x3)
        return out1 + out2 + out3


# 初始化模型和优化器
model = PolynomialRegression()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(x.unsqueeze(1))
    loss = criterion(outputs, y.unsqueeze(1))
    loss.backward()
    optimizer.step()

# 输出参数 w1、w2、w3、b 和损失
w1 = model.linear1.weight.item()
w2 = model.linear2.weight.item()
w3 = model.linear3.weight.item()
b = model.linear1.bias.item()
print("参数 w1:", w1)
print("参数 w2:", w2)
print("参数 w3:", w3)
print("参数 b:", b)
print("损失:", loss.item())

# 生成预测数据
x_pred = np.linspace(-3, 3, 100)
y_pred = w1 * x_pred + w2 * x_pred ** 2 + w3 * x_pred ** 3 + b
# 指定中文字体文件路径
font_path = r"c:\windows\fonts\SIMLI.TTF"  # 替换为你的字体文件路径

# 加载字体
fontprop = fm.FontProperties(fname=font_path, size=12)
# 绘制预测回归曲线和训练数据散点图
plt.scatter(x.numpy(), y.numpy(), label='训练数据散点', color='b', marker='o')
plt.plot(x_pred, y_pred, label='预测回归曲线', color='r')
plt.xlabel('x')
plt.ylabel('y')
plt.title('多元线性回归模型', fontproperties=fontprop)
plt.grid(True)
plt.legend(prop=fontprop)
plt.show()
