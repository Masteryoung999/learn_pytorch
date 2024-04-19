from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

img_path = "data/train/ants/6743948_2b8c096dda.jpg"
img = Image.open(img_path)

writer = SummaryWriter("logs")

# 1.transforms该如何使用
tools = transforms.ToTensor()
tensor_img = tools(img)

writer.add_image("Tensor_img", tensor_img)
writer.close()

# 2.为什么我们需要Tensor类型
