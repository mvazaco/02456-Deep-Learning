import os
import sys
import numpy as np
from tqdm.auto import tqdm

import torch

from losses import PerceptualLoss


def train2(model, optimizer, loss_fn,
          train_loader, test_loader,
          epochs=10, patience=10):

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  loss_name = loss_fn.__class__.__name__
  lr = optimizer.param_groups[0]['lr']

  out_dict = {'loss_train': [], 'loss_test': []}

  atl_test_loss = sys.maxsize
  best_loss = sys.maxsize
  countdown = patience
  
    
  for epoch in tqdm(range(epochs), unit='epoch'):
    model.train()
    train_loss = []

    for minibatch_no, (input, target, _) in tqdm(enumerate(train_loader), total=len(train_loader)):
      input, target = input.to(device), target.to(device)
      optimizer.zero_grad()
      output = model(input)

      if isinstance(loss_fn, PerceptualLoss):
        
        loss = loss_fn(output, target, input)
      else:
        loss = loss_fn(output, target)
      loss.backward()
      optimizer.step()

      train_loss.append(loss.item())
              
      model.eval()
      test_loss = []

    for input, target, original in test_loader:
      input, target, original = input.to(device), target.to(device), original.to(device)

      with torch.no_grad():
        output = model(input)
        output = output.to(device)

      if isinstance(loss_fn, PerceptualLoss):
        loss = loss_fn(output, target, input)
      else:
        loss = loss_fn(output, target)

      test_loss.append(loss.cpu().item())
      
    out_dict['loss_train'].append(np.mean(train_loss))
    out_dict['loss_test'].append(np.mean(test_loss))

    current_loss = round(loss.item(), 4)

    checkpoints_path = '/zhome/6d/e/184043/mario/DL/Checkpoints/{}_lr={}_epoch={}({})_loss={:.4f}.pth'.format(loss_name,lr,epoch+1,epochs,current_loss)
    torch.save(model.state_dict(), checkpoints_path)

  return input, target, original, output, out_dict