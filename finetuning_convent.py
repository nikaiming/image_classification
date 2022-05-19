from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms

from torchsummary import summary
from icecream import ic

transforms = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.4914, 0.48216, 0.44653),
                                                      (0.24703, 0.24349, 0.26159))])

trainset = torchvision.datasets.CIFAR10(root='./CIFAR10', train=True, download=True, transform=transforms)
testset = torchvision.datasets.CIFAR10(root='./CIFAR10', train=False, download=True, transform=transforms)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=0)

ic(trainset.data.shape)
ic(testset.data.shape)


class Net(nn.Module):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(128 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 16 * 16
        x = self.pool(F.relu(self.conv2(x)))  # 8 * 8
        x = self.pool(F.relu(self.conv3(x)))  # 4 * 4
        x = x.view(-1, 128 * 4 * 4)
        return self.fc(x)


net = Net()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=3e-4)
ic(summary(net, (3, 32, 32)))

print("Model's state_dict:")
for param_tensor in net.state_dict():
    print(param_tensor, "\t", net.state_dict()[param_tensor].size())

print(optimizer.state_dict()['param_groups'])


def train():
    """
    Training
    """
    num_epochs = 10
    since = time.time()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))

        running_loss = 0.0
        running_corrects = 0

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / trainloader.dataset.data.shape[0]
        epoch_acc = running_corrects.double() / trainloader.dataset.data.shape[0]

        print('train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
        print('-' * 10)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    save_state_dict()


def evaluation():
    """
    Evaluation
    """
    correct, total = 0, 0
    predictions = []
    net = load_state_dict()
    for i, data in enumerate(testloader, 0):
        inputs, labels = data
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        predictions.append(outputs)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('The testing set accuracy of the network is %d %%' % (100 * correct / total))


PATH = './model/finetune_model_torch.pth'


def save_state_dict():
    torch.save(net.state_dict(), PATH)


def load_state_dict():
    model = Net()
    model.load_state_dict(torch.load(PATH))
    model.eval()
    return model


def save_model(model):
    torch.save(model, PATH)


def load_model():
    model = torch.load(PATH)
    model.eval()
    return model


"""
AlexNet
"""


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))  # 最终的size
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):  # x (?, 3, 227, 227)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):  # x (?, 3, 32, 32)
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


if __name__ == "__main__":
    # train()
    # evaluation()
    alexNet = AlexNet()
    print(alexNet)
    x = torch.randn(32, 3, 227, 227)
    y = alexNet(x)
    ic(y.size())

    net = VGG('VGG11')
    print(net)
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    print(y.size())
