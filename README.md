# CNN-ResNet
CNN for image recognition
# === ver3-4  ===
import torch, time
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# === 資料預處理 ===
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

dataset = ImageFolder('/Users/baijiwei/Downloads/DL_HW1/trainingdataall', transform=transform)
num_classes = len(dataset.classes)
train_size = int(0.8 * len(dataset))
trainset, valset = random_split(dataset, [train_size, len(dataset) - train_size])
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
valloader = DataLoader(valset, batch_size=32)

# ===  ResNet ===
class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.downsample = downsample

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out)) + identity
        return self.relu(out)

class StrongerResNetV3(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.in_c = 32
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(32, blocks=2)
        self.layer2 = self._make_layer(64, blocks=3, stride=2)
        self.layer3 = self._make_layer(128, blocks=4, stride=2)
        self.layer4 = self._make_layer(128, blocks=3)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128, num_classes)

    def _make_layer(self, out_c, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_c != out_c:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_c, out_c, 1, stride, bias=False),
                nn.BatchNorm2d(out_c)
            )
        layers = [ResidualBlock(self.in_c, out_c, stride, downsample)]
        self.in_c = out_c
        layers += [ResidualBlock(out_c, out_c) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)

# === 訓練設定 ===
net = StrongerResNetV3(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

train_acc_list, val_acc_list = [], []
best_val_acc, early_stop, patience = 0, 0, 8

# === 訓練迴圈===
for epoch in range(45):
    net.train()
    correct, total = 0, 0
    for x, y in trainloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = net(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        _, pred = out.max(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    train_acc = correct / total
    train_acc_list.append(train_acc)

    # === 驗證階段 ===
    net.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in valloader:
            x, y = x.to(device), y.to(device)
            out = net(x)
            _, pred = out.max(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    val_acc = correct / total
    val_acc_list.append(val_acc)
    scheduler.step(val_acc)

    print(f"Epoch {epoch+1}/45 - Train: {train_acc:.4f}, Val: {val_acc:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(net.state_dict(), 'StrongerResNetV3.pt')
        early_stop = 0
    else:
        early_stop += 1
        if early_stop >= patience:
            print("Early stopping triggered!")
            break

# === 畫圖 ===
plt.plot(train_acc_list, label='Train acc')
plt.plot(val_acc_list, label='Val acc')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training & Validation Accuracy")
plt.legend()
plt.grid(True)
plt.show()
