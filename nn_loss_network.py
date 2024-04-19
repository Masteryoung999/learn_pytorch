import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, ReLU, Flatten, Linear, Sigmoid
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset, batch_size=64)
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()

        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            ReLU(),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            ReLU(),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            ReLU(),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10),
            Sigmoid()
        )

    def forward(self, x):
        x = self.model1(x)
        return x


loss = nn.CrossEntropyLoss()
tudui = Tudui()
for data in dataloader:
    imgs, targets = data
    outputs = tudui(imgs)
    result_loss = loss(outputs, targets)
    result_loss.backward()
    print("ok")