import utils
import os
import config2
import torch
import numpy as np

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from skimage.io import imread, imshow, concatenate_images


class AirbusDataset(Dataset):
    def __init__(self, in_df, transform=None, mode='train'):
        grp = list(in_df.groupby('ImageId'))
        self.image_ids =  [_id for _id, _ in grp] 
        self.image_masks = [m['EncodedPixels'].values for _,m in grp]
        self.transform = transform
        self.mode = mode
        self.img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])  # use mean and std from ImageNet 

    def __len__(self):
        return len(self.image_ids)
               
    def __getitem__(self, idx):
        img_file_name = self.image_ids[idx]
        if (self.mode == 'train') | (self.mode == 'validation'):
            rgb_path = os.path.join(config2.PATH_TRAIN, img_file_name)
        else:
            rgb_path = os.path.join(config2.PATH_TEST, img_file_name)
        img = imread(rgb_path)
        mask = utils.masks_as_image(self.image_masks[idx])
        
        if self.transform is not None: 
            img, mask = self.transform(img, mask)
            
        if (self.mode == 'train') | (self.mode == 'validation'):
            return self.img_transform(img), torch.from_numpy(np.moveaxis(mask, -1, 0)).float()  
        else:
            return self.img_transform(img), str(img_file_name)