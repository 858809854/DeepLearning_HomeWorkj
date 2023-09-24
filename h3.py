import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib

# 设置matplotlib正常显示中文和负号
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号
# 定义数据变换
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
# 下载MNIST数据集
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# 定义数据加载器
batch_size = 64
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# 定义Softmax分类模型
class SoftmaxClassifier(nn.Module):
    def __init__(self):
        super(SoftmaxClassifier, self).__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        x = x.view(-1, 784)  # 将图片展开成一维向量
        out = self.fc(x)
        return out


model = SoftmaxClassifier()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 10
train_losses = []
test_losses = []
train_accuracy = []
test_accuracy = []

for epoch in range(num_epochs):
    train_loss = 0.0
    correct_train = 0
    total_train = 0

    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_accuracy.append(100 * correct_train / total_train)
    train_losses.append(train_loss / len(train_loader))

    # 在测试集上测试模型
    test_loss = 0.0
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    test_accuracy.append(100 * correct_test / total_test)
    test_losses.append(test_loss / len(test_loader))

# 绘制准确率变化图
plt.figure(figsize=(10, 5))
plt.plot(range(num_epochs), train_accuracy, label='训练准确率', marker='o')
plt.plot(range(num_epochs), test_accuracy, label='测试准确率', marker='x')
plt.xlabel('迭代次数')
plt.ylabel('准确率 (%)')
plt.title('准确率随迭代次数变化')
plt.legend()
plt.grid(True)
plt.show()

# 绘制损失变化图
plt.figure(figsize=(10, 5))
plt.plot(range(num_epochs), train_losses, label='训练损失', marker='o')
plt.plot(range(num_epochs), test_losses, label='测试损失', marker='x')
plt.xlabel('迭代次数')
plt.ylabel('损失')
plt.title('损失随迭代次数变化')
plt.legend()
plt.grid(True)
plt.show()

# 计算最终分类精度和分类损失
with torch.no_grad():
    correct = 0
    total = 0
    total_loss = 0.0
    for images, labels in test_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

final_accuracy = 100 * correct / total
final_loss = total_loss / len(test_loader)

print("最终分类精度:", final_accuracy, "%")
print("最终分类损失:", final_loss)
