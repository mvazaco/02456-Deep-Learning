import torch
import torch.nn as nn
from torchvision.models import vgg, VGG11_Weights
from kornia.color import lab_to_rgb
from torchvision import transforms

import numpy as np
from dataloader import Dataprep
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class PerceptualLoss(nn.Module):
    def __init__(self, resize=False):
        super(PerceptualLoss, self).__init__()
        vgg_net = vgg.vgg11(weights=VGG11_Weights.DEFAULT).eval()
        blocks = []
        blocks.append(vgg_net.features[:4].eval())
        blocks.append(vgg_net.features[4:8].eval())
        blocks.append(vgg_net.features[8:12].eval())
        blocks.append(vgg_net.features[12:16].eval())

        # Freeze the parameters
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, gray):
        # Convert to RGB
        gray_scaled = gray * 100
        rgb_input = self.to_rgb(gray_scaled, input)
        rgb_target = self.to_rgb(gray_scaled, target)

        rgb_input = (rgb_input-self.mean) / self.std
        rgb_target = (rgb_target-self.mean) / self.std
        if self.resize:
            rgb_input = self.transform(rgb_input, mode='bilinear', size=(224, 224), align_corners=False)
            rgb_target = self.transform(rgb_target, mode='bilinear', size=(224, 224), align_corners=False)

        loss = 0.0
        x = rgb_input
        y = rgb_target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.l1_loss(x, y)
        return loss / len(self.blocks)

    def to_rgb(self, gray: torch.Tensor, ab: torch.Tensor):
        ab_scaled = (ab * 255) - 128
        lab_image = torch.cat((gray, ab_scaled), dim=1)
        rgb = lab_to_rgb(lab_image)
        return rgb

class TVLoss(nn.Module):
    def forward(self, input, target):
        """
        Compute the Total Variation Loss.
        Inputs:
        - img: PyTorch tensor of shape (N, 2, H, W) holding an input image.
        Returns:
        - loss: Scalar giving the mean variation loss for img.
        """
        h_var = input[:, :, :-1, :-1] - input[:, :, 1:, :-1]
        w_var = input[:, :, :-1, :-1] - input[:, :, :-1, 1:]
        diff = h_var**2 + w_var**2
        return diff.mean()


class WeightedColorLoss(nn.Module):
    def __init__(self, dataloader, num_bins=10):
        super(WeightedColorLoss, self).__init__()
        self.num_bins = num_bins
        self.criterion = nn.SmoothL1Loss(reduction='none')  # Using L1 Loss as an example
        # Calculate the bin edges and weights during initialization
        self.precompute_bins_and_weights(dataloader, num_bins)
        print(self.weights_a, self.weights_a)

    def precompute_bins_and_weights(self, dataloader, num_bins):
        # Flatten the ab channels in the dataset and compute histogram bin edges
        print("Precomputing color weights...")
        a_values, b_values = [], []
        for _, target, _ in dataloader:
            a_values.append(target[:, 0, :, :].flatten())
            b_values.append(target[:, 1, :, :].flatten())
        a_values = np.concatenate(a_values)
        b_values = np.concatenate(b_values)

        # Compute histogram bin edges
        a_edges = np.linspace(a_values.min(), a_values.max(), num=num_bins + 1)
        b_edges = np.linspace(b_values.min(), b_values.max(), num=num_bins + 1)

        # Compute the weights for each bin
        a_counts, a_bins = np.histogram(a_values, bins=a_edges)
        b_counts, b_bins = np.histogram(b_values, bins=b_edges)
        
        # Prevent division by zero
        a_counts[a_counts == 0] = 1
        b_counts[b_counts == 0] = 1
        
        # Calculate class weights as the inverse of the frequencies
        a_weights = 1.0 / np.sqrt(a_counts)
        b_weights = 1.0 / np.sqrt(b_counts)
        
        # Normalize weights to sum to the number of bins
        a_weights /= a_weights.sum() / (num_bins)
        b_weights /= b_weights.sum() / (num_bins)

        self.register_buffer("bin_edge_a", torch.Tensor(a_edges))
        self.register_buffer("bin_edge_b", torch.Tensor(b_edges))
        self.register_buffer("weights_a", torch.FloatTensor(a_weights))
        self.register_buffer("weights_b", torch.FloatTensor(b_weights))

    def forward(self, input, target):
        # Assign each pixel in the input and target to a bin
        a_target_indices = torch.bucketize(target[:, 0, :, :].contiguous(), self.bin_edge_a) - 1
        b_target_indices = torch.bucketize(target[:, 1, :, :].contiguous(), self.bin_edge_b) - 1

        # Clamp the indices to ensure they are within the valid range
        a_target_indices = torch.clamp(a_target_indices, min=0, max=self.num_bins - 1)
        b_target_indices = torch.clamp(b_target_indices, min=0, max=self.num_bins - 1)

        # Gather the weights for each pixel from the precomputed weights
        a_weights = self.weights_a[a_target_indices]
        b_weights = self.weights_b[b_target_indices]

        # Compute the weighted loss
        a_loss = self.criterion(input[:, 0, :, :], target[:, 0, :, :]) * a_weights
        b_loss = self.criterion(input[:, 1, :, :], target[:, 1, :, :]) * b_weights
        loss = (a_loss + b_loss).mean()  # Mean over the batch

        return loss