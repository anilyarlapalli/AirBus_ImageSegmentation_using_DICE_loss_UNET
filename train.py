import config2
import pandas as pd
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import transformations
import dataset
import torch
import losses
import torch.nn as nn
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
import metrics
import utils
import engine
import torch.optim as optim
import model_unet_mini
import losses_ref

unique_img_ids = config2.UNIQUE_IMG_IDS

train_ids, val_ids = train_test_split(unique_img_ids, test_size=0.05, stratify=unique_img_ids['counts'], random_state=42)
train_df = pd.merge(config2.MASKS, train_ids)
valid_df = pd.merge(config2.MASKS, val_ids)

print((train_df['ImageId']).value_counts().shape)

img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

train_transform = transformations.DualCompose([transformations.HorizontalFlip(), 
                                               transformations.VerticalFlip(),
                                               transformations.RandomCrop((256,256,3))])

val_transform = transformations.DualCompose([transformations.CenterCrop((512,512,3))])

train_dataset = dataset.AirbusDataset(train_df[:2100], transform=train_transform, mode='train')
val_dataset = dataset.AirbusDataset(valid_df[:200], transform=val_transform, mode='validation')

print('Train samples : %d | Validation samples : %d' % (len(train_dataset), len(val_dataset)))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config2.BATCH_SZ_TRAIN, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config2.BATCH_SZ_VALID, shuffle=False, num_workers=0)

run_id = 1

def loss_fn(LOSS):   
    if LOSS == 'BCEWithDigits':
        criterion = nn.BCEWithLogitsLoss()
    elif LOSS == 'FocalLossWithDigits':
        criterion = losses.MixedLoss(10, 2)
    elif LOSS == 'BCEDiceWithLogitsLoss':
        criterion = losses.BCEDiceWithLogitsLoss()
    elif LOSS == 'BCEJaccardWithLogitsLoss':
        criterion = losses.BCEJaccardWithLogitsLoss()
    else:
        raise NameError("loss not supported")
    
    return criterion

net = model_unet_mini.ImageSegmentation(num_classes=1)

engine.train(init_optimizer=lambda lr: optim.Adam(net.parameters(), lr=lr),
        lr = config2.LR,
        n_epochs = config2.N_EPOCHS,
        model=net,
        criterion= losses_ref.IoULoss(),         # loss_fn('BCEDiceWithLogitsLoss'),
        train_loader=train_loader,
        valid_loader=val_loader,
        train_batch_sz = config2.BATCH_SZ_TRAIN,
        valid_batch_sz = config2.BATCH_SZ_VALID,
        fold=run_id
        )

