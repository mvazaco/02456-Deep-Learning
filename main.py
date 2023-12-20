import os
import json
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# Our imports
from dataloader import Dataprep
from models import ColorizationNet
from train import train
from train2 import train2
from utils import plot_metrics, plot_outputs_and_targets
from losses import PerceptualLoss, WeightedColorLoss, TVLoss


def main():
    if torch.cuda.is_available():
        print("The code will run on GPU.")
    else:
        print("The code will run on CPU.")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_options = {
        "ColorizationNet": ColorizationNet
    }
    optimizer_options = {
        "SGD": torch.optim.SGD,
        "Adam": torch.optim.Adam,
        "AdamW": torch.optim.AdamW
    }
    loss_options = {
        "MSE": nn.MSELoss,
        "PerceptualLoss": PerceptualLoss,
        "WeightedColorLoss": WeightedColorLoss,
        "TotalVarianceLoss": TVLoss,
        "L1": nn.L1Loss,
        "L1Smooth": nn.SmoothL1Loss
    }
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=model_options.keys(), default="ColorizationNet")
    parser.add_argument("--optimizer", type=str, choices=optimizer_options.keys(), default="Adam")
    parser.add_argument("--loss", type=str, choices=loss_options.keys(), default="MSE")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--aug", type=int, default=1)
    args = parser.parse_args()

    #Load Data
    size = 256
    if args.aug == 1:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.5, contrast=0.5),
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])

    data_path = '/zhome/6d/e/184043/mario/DL/dataset/'  
    data_train = Dataprep(transform, data_path=data_path, train=True)
    data_test  = Dataprep(transform, data_path=data_path, train=False)

    batch_size = 128
    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader  = DataLoader(data_test, batch_size=batch_size, shuffle=False, num_workers=8)

    # Model
    model = model_options[args.model]().to(device)
    # Optimizer
    optimizer = optimizer_options[args.optimizer](model.parameters(), lr=args.lr)
    # Loss function
    if args.loss == "WeightedColorLoss":
        loss_fn = loss_options[args.loss](train_loader).to(device)
    else:
        loss_fn = loss_options[args.loss]().to(device)

    input, target, original, output, out_dict = train(model=model, optimizer=optimizer, loss_fn=loss_fn,
                                       train_loader=train_loader, test_loader=test_loader,
                                       epochs=args.epochs, patience=10)


    path = '/zhome/6d/e/184043/mario/DL/Results/out_dicts/{}_{}_{}_lr={}_epochs={}_aug={}.json'
    path = path.format(args.model,args.optimizer,args.loss,args.lr,args.epochs,args.aug)
    with open(path, 'w') as json_file:
        json.dump(out_dict, json_file)

    figure = plot_outputs_and_targets(input, output, original)
    path = '/zhome/6d/e/184043/mario/DL/Results/figs/rgb_reconstructions/{}_{}_{}_lr={}_epochs={}_aug={}.png'
    path = path.format(args.model,args.optimizer,args.loss,args.lr,args.epochs,args.aug)
    figure.savefig(path)
        
if __name__ == main():
    main()
