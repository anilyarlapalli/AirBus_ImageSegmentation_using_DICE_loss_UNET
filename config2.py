import os
import pandas as pd

DATA_ROOT = r'input'
PATH_TRAIN = os.path.join(DATA_ROOT,'train_v2')
PATH_TEST = os.path.join(DATA_ROOT,'test_v2')

# Booleans
SHOW_PIXELS_DIST = False
SHOW_SHIP_DIAG = False
SHOW_IMG_LOADER = False

# Training variables
BATCH_SZ_TRAIN = 16
BATCH_SZ_VALID = 4
LR = 1e-4
N_EPOCHS = 8

# Define loss function
LOSS = 'BCEWithDigits' # BCEWithDigits | FocalLossWithDigits | BCEDiceWithLogitsLoss | BCEJaccardWithLogitsLoss

MASKS = pd.read_csv(os.path.join(DATA_ROOT, 'train_ship_segmentations_v2.csv'))
DF_WSHIPS = MASKS.dropna()
DF_WSHIPS = DF_WSHIPS.groupby('ImageId').size().reset_index(name='counts')
DF_WOSHIPS = MASKS[MASKS['EncodedPixels'].isna()]

UNIQUE_IMG_IDS = MASKS.groupby('ImageId').size().reset_index(name='counts')
