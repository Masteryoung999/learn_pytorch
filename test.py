import torch
import torchvision
from PIL import Image
from torch import nn

device = torch.device("mps")
images_path = "./images/airplane.png"
image = Image.open(images_path)
print(image)
image = image.convert('RGB')

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])

image = transform(image)
image = image.to(device)
print(image.shape)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.model(x)
        return x

model = torch.load("tudui_31.pth", map_location=torch.device('mps'))
print(model)
image = torch.reshape(image, (1, 3, 32, 32))

model.eval()
with torch.no_grad():
    output = model(image)
print(output)

print(output.argmax(1))