import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import lab2rgb, rgb2lab, rgb2gray

import torch
import torch.nn.functional as F


def recontruct_rgb(input, output):
  lab_image = torch.cat((input, output), dim=0).cpu().numpy()
  lab_image = lab_image.transpose((1, 2, 0))
  lab_image[:, :, 0:1] = lab_image[:, :, 0:1] * 100
  lab_image[:, :, 1:3] = lab_image[:, :, 1:3] * 255 - 128
  rgb_image = lab2rgb(lab_image.astype(np.float64))
  return rgb_image


def plot_outputs_and_targets(input, output, original):
  figure = plt.figure()
  for k in range(5):
    plt.subplot(2, 5, k+1)
    output_rgb = recontruct_rgb(input[k], output[k])
    plt.imshow(output_rgb)
    plt.title('Predicted')
    plt.axis('off')

    plt.subplot(2, 5, k+5+1)
    plt.imshow(original[k].cpu().numpy().transpose(1, 2, 0))
    plt.title('Original')
    plt.axis('off')
  
  plt.tight_layout()
  plt.show()
  return figure


def plot_metrics(data):
  num_epochs = range(1, len(data['loss_train'])+1)
  
  plt.figure()
  plt.plot(num_epochs, data['loss_train'],      color='blue',   label='train')
  plt.plot(num_epochs, data['loss_test'], '--', color='blue',   label='test')
  plt.xlabel('Epochs', fontsize=15)
  plt.ylabel('Loss', fontsize=15)
  plt.xticks(range(1, len(data['loss_train'])+1), fontsize=12)
  plt.yticks(fontsize=12)
  # plt.ylim([0.4, 1])
  plt.legend(fontsize=15)
  plt.grid()

  plt.tight_layout()
  plt.show()
  return