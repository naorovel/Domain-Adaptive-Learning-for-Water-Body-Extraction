import os  # If directory manipulation is involved (e.g., path joining, existence checks)
import torch  # For tensor operations
import numpy as np  # For numerical operations like stacking arrays
from torch.utils.data import Dataset  # Base class for custom PyTorch datasets
import rasterio  # For reading .tif files
from rasterio.merge import merge
from rasterio.windows import Window
import glob
from rasterio.crs import CRS
from rasterio.transform import from_origin
import random
from os.path import basename
from rasterio.transform import Affine
from FDA.FDA import apply_fda_and_save
import shutil
from satellite_data import SatelliteData

feature_dir_train = "../data/CN/feature_trimmed.tif"
label_dir_train = "../data/CN/label_trimmed.tif"
feature_tiles_train = "../data/CN/tiles/features/"
label_tiles_train = "../data/CN/tiles/labels/"
feature_tiles_mergeback_train = "../data/CN/tiles/merge/merged_feature.tif"
label_tiles_mergeback_train = "../data/CN/tiles/merge/merged_label.tif"
random_sample_train_dir = "../data/CN/random_samples"
processed_train_dir = "../data/CN/processed/train"

processed_val_dir = "../data/CN/processed/validation"
processed_test_dir = "../data/BZ/processed/test"

# Test dataset
feature_dir_test = "../data/BZ/feature.tif"
label_dir_test = "../data/BZ/label.tif"
feature_tiles_test = "../data/BZ/tiles/features"
label_tiles_test = "../data/BZ/tiles/labels"
feature_tiles_mergeback_test = "../data/BZ/tiles/merge/merged_feature.tif"
label_tiles_mergeback_test = "../data/BZ/tiles/merge/merged_label.tif"

train_data: SatelliteData
validation_data: SatelliteData
test_data: SatelliteData

def get_train_data(): 
    print("Generating training data...")
    global train_data
    train_data = SatelliteData(feature_dir_train, 
                              label_dir_train, 
                              feature_tiles_train,
                              label_tiles_train,
                              feature_tiles_mergeback_dir=feature_tiles_mergeback_train,
                              label_tiles_mergeback_dir=label_tiles_mergeback_train,
                              random_sample_dir=random_sample_train_dir,
                              random_sample=True
                              )
    
    # train_data.save_train_baseline(processed_train_dir)
    
    print("Finished generating training data.")
    return train_data

def get_validation_data(val_ratio):
    ## ASSUME THAT TRAINING DATA HAS ALREADY BEEN CREATED
    print("Generating validation data...")
    global validation_data
    validation_data = train_data.create_validation_set(val_ratio) 

    return validation_data

    # train_data.get_batches(16)
    # next_batch = train_data.get_next_batch()

def get_test_data(): 
    print("Generating test data...")
    global test_data
    test_data = SatelliteData(feature_dir_test, 
                              label_dir_test, 
                              feature_tiles_test,
                              label_tiles_test,
                              feature_tiles_mergeback_dir=feature_tiles_mergeback_test,
                              label_tiles_mergeback_dir=label_tiles_mergeback_test,
                              random_sample = False,
                              ) 


def main(): 
    get_train_data()
    get_validation_data(0.1)
    #get_test_data()

if __name__ == "__main__": 
    main()
