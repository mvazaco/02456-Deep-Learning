import os
import glob
import numpy as np
from PIL import Image
import torch
from skimage import io, color

from skimage.color import rgb2lab, rgb2gray

class Dataprep(torch.utils.data.Dataset):
  def __init__(self, transform, data_path, train=True):
    self.transform = transform
    data_path = os.path.join(data_path, 'train/class' if train else 'val/class')
    self.path = data_path
    self.image_paths = glob.glob(data_path + '/*.jpg')
      
  def __len__(self):
    return len(self.image_paths)
  
  def _set_seed(self, seed):
    random.seed(seed)
    torch.manual_seed(seed)

  def __getitem__(self, idx):
    image_path = self.image_paths[idx]
    image = Image.open(image_path)
    image = image.convert('RGB')
    image = self.transform(image)
    original = image
    image = image.permute(1, 2, 0).numpy()
      
    X = rgb2gray(image)
    X = torch.from_numpy(X).unsqueeze(0).float()

    Y = rgb2lab(image)
    Y = (Y + 128) / 255
    Y = Y[:, :, 1:3]
    Y = torch.from_numpy(Y.transpose((2, 0, 1))).float()

    return X, Y, original