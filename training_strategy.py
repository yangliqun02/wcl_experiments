import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import csv
import os

# 创建文件夹用于保存结果
# output_dir = 'training_verification'
output_dir = 'training_verification_adam'
os.makedirs(output_dir, exist_ok=True)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超参数
batch_size = 128
lr = 0.1
momentum = 0.9
weight_decay = 5e-4
log_interval = 50  # 每50个batch记录一次

# 数据预处理
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 加载数据集
train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# 构建 ResNet-18 并适配 CIFAR-10
model = torchvision.models.resnet18(weights=None)
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()
model.fc = nn.Linear(512, 10)
model = model.to(device)

# 保存初始权重
initial_weights_path = os.path.join(output_dir, 'initial.pth')
torch.save(model.state_dict(), initial_weights_path)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# 学习率调度器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# 创建CSV文件并写入表头
log_file = os.path.join(output_dir, 'training_log.csv')
with open(log_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Step', 'Epoch', 'Batch', 'Train Loss', 'Test Accuracy'])

# 用于评估的函数
def evaluate(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return correct / total

# 开始训练
print("开始训练...")

global_step = 0
total_epochs = 100  

for epoch in range(total_epochs):
    model.train()
    train_loss = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        global_step += 1

        # 每50个batch记录一次loss和准确率
        if (batch_idx + 1) % log_interval == 0:
            avg_loss = train_loss / log_interval
            test_acc = evaluate(model, test_loader)
            print(f"Epoch: {epoch+1} | Batch: {batch_idx+1} | Loss: {avg_loss:.4f} | Test Acc: {test_acc*100:.2f}%")

            # 写入CSV
            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([global_step, epoch + 1, batch_idx + 1, avg_loss, test_acc])

            train_loss = 0  # 重置loss计数器

    scheduler.step()

    # 在第4个epoch后保存中间权重
    if epoch == 50: 
        mid_weights_path = os.path.join(output_dir, 'mid.pth')
        if not os.path.exists(mid_weights_path):
            torch.save(model.state_dict(), mid_weights_path)
            print("中间模型已保存。")

# 最终模型
final_weights_path = os.path.join(output_dir, 'final.pth')
torch.save(model.state_dict(), final_weights_path)
print("训练完成，最终权重已保存。")