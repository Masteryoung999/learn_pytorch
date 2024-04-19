from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")
img_path = "data/train/ants/6743948_2b8c096dda.jpg"
img_PIL = Image.open(img_path)
img_array = np.array(img_PIL)


writer.add_image("train", img_array, 1, dataformats='HWC')

writer.close()