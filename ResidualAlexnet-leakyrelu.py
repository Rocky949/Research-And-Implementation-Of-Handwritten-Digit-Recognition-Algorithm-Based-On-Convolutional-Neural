# 温馨提示：运行本程序训练需要5.2GB的GPU显存
# 如果显存不够用可以减少Batchsize 32，16都可以减少资源的使用
# 程序可以运行 实际准确率98%以上
import time
import os
import numpy as np

import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn


from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

from PIL import Image
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Batch_size = 64
custom_transform1 = transforms.Compose([transforms.Resize([64, 64]),
                                        transforms.ToTensor()])
train_dataset = torchvision.datasets.MNIST(
    root='./MNIST',
    train=True,
    download=True,
    transform=custom_transform1
)
test_dataset = torchvision.datasets.MNIST(
    root='./MNISt',
    train=False,
    download=True,
    transform=custom_transform1
)
# define train loader
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    shuffle=True,
    batch_size=Batch_size
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    shuffle=True,
    batch_size=Batch_size
)

#搭建Alexnet网络模型
class ResidualAlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResidualAlexNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.LeakyReLU(inplace=True)  # Replace ReLU with LeakyReLU
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(192)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(384)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.LeakyReLU(inplace=True),  # Replace ReLU with LeakyReLU
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.LeakyReLU(inplace=True),  # Replace ReLU with LeakyReLU
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        x = self.pool(self.relu(residual))
        residual = self.conv2(x)
        residual = self.bn2(residual)
        x = self.pool(self.relu(residual))
        residual = self.conv3(x)
        residual = self.bn3(residual)
        x = self.relu(residual)
        residual = self.conv4(x)
        residual = self.bn4(residual)
        x = self.relu(residual)
        residual = self.conv5(x)
        residual =residual +x
        residual = self.bn5(residual)
        x = self.pool(self.relu(residual))
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        logits = self.classifier(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas




# 实例化模型并打印展示
net = ResidualAlexNet(10)
print(net)
print(net(torch.randn([1,1,64,64])))

#设定训练的轮次
NUM_EPOCHS = 10

model = ResidualAlexNet(num_classes=10)

model = model.to(DEVICE)


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

valid_loader = test_loader


def compute_accuracy_and_loss(model, data_loader, device):
    correct_pred, num_examples = 0, 0
    cross_entropy = 0.
    for i, (features, targets) in enumerate(data_loader):
        features = features.to(device)
        targets = targets.to(device)

        logits, probas = model(features)
        cross_entropy += F.cross_entropy(logits, targets).item()
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float() / num_examples * 100, cross_entropy / num_examples


start_time = time.time()
train_acc_lst, valid_acc_lst = [], []
train_loss_lst, valid_loss_lst = [], []

#训练阶段
for epoch in range(NUM_EPOCHS):

    model.train()

    for batch_idx, (features, targets) in enumerate(train_loader):

        ### PREPARE MINIBATCH
        features = features.to(DEVICE)
        targets = targets.to(DEVICE)

        ### FORWARD AND BACK PROP
        logits, probas = model(features)
        cost = F.cross_entropy(logits, targets)
        optimizer.zero_grad()

        cost.backward()

        ### UPDATE MODEL PARAMETERS
        optimizer.step()

        ### LOGGING
        if not batch_idx % 120:
            print(f'Epoch: {epoch + 1:03d}/{NUM_EPOCHS:03d} | '
                  f'Batch {batch_idx:03d}/{len(train_loader):03d} |'
                  f' Cost: {cost:.4f}')

    # no need to build the computation graph for backprop when computing accuracy
    model.eval()
    with torch.set_grad_enabled(False):
        train_acc, train_loss = compute_accuracy_and_loss(model, train_loader, device=DEVICE)
        valid_acc, valid_loss = compute_accuracy_and_loss(model, valid_loader, device=DEVICE)
        train_acc_lst.append(train_acc)
        valid_acc_lst.append(valid_acc)
        train_loss_lst.append(train_loss)
        valid_loss_lst.append(valid_loss)
        print(f'Epoch: {epoch + 1:03d}/{NUM_EPOCHS:03d} Train Acc.: {train_acc:.2f}%'
              f' | Validation Acc.: {valid_acc:.2f}%')

    elapsed = (time.time() - start_time) / 60
    print(f'Time elapsed: {elapsed:.2f} min')

elapsed = (time.time() - start_time) / 60
print(f'Total Training Time: {elapsed:.2f} min')

#训练损失和测试损失关系图
plt.plot(range(1, NUM_EPOCHS+1), train_loss_lst, label='Training loss')
plt.plot(range(1, NUM_EPOCHS+1), valid_loss_lst, label='Validation loss')
plt.legend(loc='upper right')
plt.ylabel('Cross entropy')
plt.xlabel('Epoch')
plt.show()

# 保存模型
torch.save(model.state_dict(), "ResidualAlexNet-LeakyRelu.pth")

# 加载模型
model.load_state_dict(torch.load("ResidualAlexNet-LeakyRelu.pth"))

#模型测试与评估
model.eval()
with torch.set_grad_enabled(False): # save memory during inference
    test_acc, test_loss = compute_accuracy_and_loss(model, test_loader, DEVICE)
    print(f'Test accuracy: {test_acc:.2f}%')

test_loader = DataLoader(dataset=train_dataset,
                         batch_size=64,
                         shuffle=True)

for features, targets in test_loader:
    break

_, predictions = model.forward(features[:8].to(DEVICE))
predictions = torch.argmax(predictions, dim=1)

#展示结果图
print(features.size())
fig = plt.figure()
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.tight_layout()
    plt.imshow(features[i][0],interpolation='none')
    plt.title("Prediction: {}".format(predictions[i]))
plt.show()
