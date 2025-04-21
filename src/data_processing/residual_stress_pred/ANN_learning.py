# 导入必要的PyTorch库
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 设置随机种子保证结果可复现
torch.manual_seed(42)

# 1. 准备训练数据（这里我们使用简单的示例数据）
# 假设我们的任务是学习输入到输出的简单映射：输入是5维向量，输出是2维向量
# 示例数据：输入为5个特征，输出为2个目标值
X_train = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5],
                        [0.5, 0.4, 0.3, 0.2, 0.1],
                        [0.9, 0.8, 0.7, 0.6, 0.5],
                        [0.5, 0.6, 0.7, 0.8, 0.9]], dtype=torch.float32)

y_train = torch.tensor([[1, 2],
                        [5, 4],
                        [9, 8],
                        [5, 6]], dtype=torch.float32)


# 2. 创建自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# 创建数据集和数据加载器
dataset = CustomDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)


# 3. 定义神经网络模型
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # 定义网络结构
        self.fc1 = nn.Linear(5, 8)  # 输入层（5个特征）到隐藏层（8个神经元）
        self.fc2 = nn.Linear(8, 2)  # 隐藏层（8个神经元）到输出层（2个输出）
        self.relu = nn.ReLU()  # 激活函数

    def forward(self, x):
        x = self.fc1(x)  # 全连接层1
        x = self.relu(x)  # 应用激活函数
        x = self.fc2(x)  # 全连接层2（输出层）
        return x


# 4. 初始化模型、损失函数和优化器
model = SimpleNN()
criterion = nn.MSELoss()  # 均方误差损失函数（适用于回归问题）
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Adam优化器，学习率0.01

# 5. 训练循环
num_epochs = 1000  # 训练轮数

for epoch in range(num_epochs):
    for inputs, labels in dataloader:
        # 前向传播
        outputs = model(inputs)

        # 计算损失
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()  # 清空之前的梯度
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新参数

    # 每隔100个epoch打印一次损失
    if (epoch + 1) % 100 == 0:
        # 计算整个训练集的损失
        with torch.no_grad():  # 不需要计算梯度
            total_loss = 0
            for inputs, labels in dataloader:
                outputs = model(inputs)
                total_loss += criterion(outputs, labels)
            avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss.item():.4f}')

# 6. 测试模型
# 使用训练好的模型进行预测
test_input = torch.tensor([0.3, 0.4, 0.5, 0.6, 0.7], dtype=torch.float32)
model.eval()  # 设置模型为评估模式
with torch.no_grad():
    prediction = model(test_input)
    print(f"\n测试输入: {test_input.numpy()}")
    print(f"模型预测结果: {prediction.numpy()}")

# 7. 查看模型参数（可选）
print("\n模型结构：")
print(model)

print("\n第一层权重：")
print(model.fc1.weight)