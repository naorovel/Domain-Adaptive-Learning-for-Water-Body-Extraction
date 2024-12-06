import torch
import os
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.functional import  binary_cross_entropy_with_logits, sigmoid
import pytorch_lightning as pl
import torchmetrics
from torchmetrics import Metric
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import segmentation_models_pytorch as smp
from tif_processor import SatelliteDataset
from metrics import MIoU
from loss import dice_bce_loss_with_logits, dice_loss_with_logits
from prepare_data import *
from torch.utils.data import random_split, DataLoader

batch_size=4
shuffle=True
EPOCHS=5
val_ratio = 0.1

train_data = get_test_data()
val_data = get_validation_data(val_ratio)
test_data = get_test_data()

train_loader = DataLoader(train_data, batch_size, shuffle)
val_loader = DataLoader(val_data, batch_size, shuffle)
test_loader = DataLoader(test_data, batch_size, shuffle)

def mkpath(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)