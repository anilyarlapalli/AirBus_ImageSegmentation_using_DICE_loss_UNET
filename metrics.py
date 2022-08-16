import torch
import numpy as np


def compute_metrics(pred, true, batch_size=16, threshold=0.5):
    pred = pred.view(batch_size, -1)
    true = true.view(batch_size, -1)
    
    pred = (pred > threshold).float()
    true = (true > threshold).float()
    
    pred_sum = pred.sum(-1)
    true_sum = true.sum(-1)
    
    neg_index = torch.nonzero(true_sum == 0)
    pos_index = torch.nonzero(true_sum >= 1)
    
    dice_neg = (pred_sum == 0).float()
    dice_pos = 2 * ((pred * true).sum(-1)) / ((pred + true).sum(-1))
    
    dice_neg = dice_neg[neg_index]
    dice_pos = dice_pos[pos_index]
    
    dice = torch.cat([dice_pos, dice_neg])
    jaccard = dice / (2 - dice)
    
    return dice.numpy(), jaccard.numpy()

class metrics:
    def __init__(self, batch_size=16, threshold=0.5):
        self.threshold = threshold
        self.batchsize = batch_size
        self.dice = []
        self.jaccard = []
    def collect(self, pred, true):
        pred = torch.sigmoid(pred)
        dice, jaccard = compute_metrics(pred, true, batch_size=self.batchsize, threshold=self.threshold)
        self.dice.extend(dice)
        self.jaccard.extend(jaccard)
    def get(self):
        #print(self.dice.shape, self.dice)
        dice = np.nanmean(self.dice)
        jaccard = np.nanmean(self.jaccard)
        return dice, jaccard