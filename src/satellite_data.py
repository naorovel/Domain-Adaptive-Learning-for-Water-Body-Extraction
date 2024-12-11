"""
THIS FILE IS NO LONGER IN USE
"""

import math
import torch  # For tensor operations
from torch.utils.data import Dataset  # Base class for custom PyTorch datasets
from tif_utils import *
import random

class SatelliteData(Dataset):
    """
    Dataset class. Convert tif files (tiles) into a dataset for model training/testing.
    """
    
    mu: torch.float16
    sigma: torch.float16
    feature_tiles = []
    label_tiles = []
    patches = []
    target_dir = "../data/style"
    validation_dataset: any = None
    validation_dataset_dir = ""
    batches = []
    style_dir = "../data/style/tiles"

    def __init__(self, feature_dir, label_dir,
                 feature_tile_dir, label_tile_dir,
                 feature_tiles_mergeback_dir=None, 
                 label_tiles_mergeback_dir=None,
                 random_sample_dir=None,
                 random_sample=False,
                 processed_patch_dir = None,
                 split_size = 512,
                 patch_size = 256, 
                 num_samples=9,
                 prob = {
                 'p_FDA': 0.75,
                 'p_hflip': 0.5,
                 'p_vflip': 0.5,
                 'p_90rot': 0.25,
                 'p_270rot': 0.25},
                 validation_dataset = False):
        self.feature_dir = feature_dir
        self.label_dir = label_dir
        self.feature_tile_dir = feature_tile_dir
        self.label_tile_dir = label_tile_dir
        self.feature_tiles_mergeback_dir = feature_tiles_mergeback_dir
        self.label_tiles_mergeback_dir = label_tiles_mergeback_dir
        self.random_sample_dir = random_sample_dir
        self.random_sample = random_sample
        self.processed_patch_dir = processed_patch_dir
        self.split_size = split_size
        self.patch_size = patch_size
        self.num_samples = num_samples
        self.prob = prob

        if not self.empty_dir(self.feature_tile_dir) and not self.empty_dir(self.label_tile_dir) and not validation_dataset: 
            if "Y".lower() == input("Original split tiles in this directory already exist. Would you like to regenerate the dataset? [Y/n] ").lower():
                self.generate_split_tiles()
                self.read_split_tiles()
                self.feature_tiles = self.normalize_tiles(self.feature_tiles) # TODO: Should calculate mean and standard deviation for each tile - put this in the data loader - this is more "centering" the data
        
            if random_sample:
                self.generate_patches()
            else: 
                print("Generated patches already...")

    def get_patch_dirs(self): 
        patch_dirs = []
        for root, dirs, files in os.walk(self.random_sample_dir + "/features"):
            for name in files: 
                dirname = root.split(os.path.sep)[-1]
                patch_dirs.append(os.path.sep + dirname + os.sep + name)

        return patch_dirs
    
    def save_train_baseline(self, dir): 
        patch_list = self.get_patch_dirs()
        for patch in patch_list:
            filename = patch[5:]
            transform_patch_old(self.random_sample_dir, patch, filename, self.prob, save_dir=dir)

    def get_batches(self, batch_size):
        print(f"Creating batches of size {batch_size}...")
        # Returns batches of tiles as filenames to use for epochs
        patch_list = self.get_patch_dirs()
        num_batches = math.ceil(len(patch_list)/batch_size)
        batch_list = np.array_split(patch_list, num_batches) 
        self.batches = batch_list
        return batch_list  
    
    def get_batch_style_tiles(self, num): 
        water_tiles = os.listdir(self.style_dir+os.path.sep+"water")
        water_tiles = [self.style_dir+os.path.sep+"water"+os.path.sep+name for name in water_tiles]

        general_tiles = os.listdir(self.style_dir + os.path.sep + "general")
        general_tiles = [self.style_dir+os.path.sep+"general"+os.path.sep+name for name in general_tiles]
        
        all_tiles = water_tiles + general_tiles

        random_style_tiles = random.sample(all_tiles, num)
        
        return random_style_tiles

    def get_next_batch(self):
        print(f"Getting next batch...")
        curr_batch = self.batches.pop()
        processed_batch = []
        style_tiles = self.get_batch_style_tiles(len(curr_batch))
        for i in range(len(curr_batch)): 
            print(f"Transforming TIF of {curr_batch[i]}...")
            processed_batch.append(transform_patch_old(self.random_sample_dir, curr_batch[i], style_tiles[i], self.prob, self.processed_patch_dir, fda_path=style_tiles[i]))
        return processed_batch

    def generate_split_tiles(self):
        if not self.empty_dir(self.feature_tile_dir) or not self.empty_dir(self.label_tile_dir):
            if "Y".lower() == input("Tiles in this directory already exist. Would you like to split the input TIF again? [Y/n] ").lower():
                print("Existing tiles will be overwritten.")
                clear_dir(self.feature_tile_dir)
                clear_dir(self.label_tile_dir)
                read_and_split_tif(self.feature_dir, self.feature_tile_dir, self.split_size)
                read_and_split_tif(self.label_dir, self.label_tile_dir, self.split_size)
                print(f"New tiles of size {self.split_size} were created.")
                self.verify_split()
            else: 
                print("Existing tiles will be used.")
        else: 
            read_and_split_tif(self.feature_dir, self.feature_tile_dir, self.split_size)
            read_and_split_tif(self.label_dir, self.label_tile_dir, self.split_size)
            print(f"New tiles of size {self.split_size} were created.")
            print(f"The dataset contains {len(os.listdir(self.feature_tile_dir))} tiles of size {self.split_size}x{self.split_size}." )
            self.verify_split()

    def verify_split(self): 
        print("Verifying split against merged TIF...")
        original_width, original_height = get_tif_size(self.feature_dir)
            
        merge_tiles_to_tif(self.feature_tile_dir, 
                           self.feature_tiles_mergeback_dir, 
                           original_width,
                           original_height,
                           self.split_size)
        merge_tiles_to_tif(self.label_tile_dir, 
                           self.label_tiles_mergeback_dir, 
                           original_width,
                           original_height,
                           self.split_size) 
            
        are_same, message = compare_tif_images(self.feature_dir, self.feature_tiles_mergeback_dir)
        if are_same: 
            print("TIF file was split correctly. ")
        else: 
            print("TIF file was not split correctly. Details below: \n")
            print(message)

    def read_split_tiles(self): 
        print("Reading split tiles...")
        self.feature_tiles = read_files(self.feature_tile_dir)
        self.label_tiles = read_files(self.label_tile_dir)
        print(f"Finished reading {len(self.feature_tiles)} split tiles.")

        assert len(self.feature_tiles) == len(self.label_tiles), "Error: Number of features does not match number of masks!"


    def normalize_tiles(self, tiles):
        print("Normalizing tiles...")
        mu = torch.mean(tiles, (0, 2, 3)).reshape((1, -1, 1, 1))
        sigma = torch.std(tiles, (0, 2, 3)).reshape((1, -1, 1, 1))
        normalize(tiles, mu, sigma)


    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return (self.features[idx,:,:,:], self.masks[idx,:,:,:])

    def empty_dir(self, dir): 
        return len(os.listdir(dir)) == 0

    def generate_patches(self):
        print("Generating patches...")
        clear_dir(self.random_sample_dir)
        random_samples_from_tiles(
            self.feature_tile_dir, self.label_tile_dir, 
            self.random_sample_dir, self.patch_size, 
            self.num_samples)
        num_patches = sum([len(files) for r, d, files in os.walk(self.random_sample_dir+"/features")])
        print(f"Generated {num_patches} patches.")

    # def apply_fda(self, water): 
    #     """
    #     Assume that output directory has been cleaned/is empty (except for .gitkeep). 
    #     """
    #     print(f"Processing patches with FDA with p={self.prob['p_FDA']}.")
    #     fda, no_fda = process_tiles_with_fda(self.random_sample_dir+"/features", self.random_sample_dir + "/features", self.target_dir, len(os.listdir(self.random_sample_dir)), apply_probability=self.prob["p_FDA"])

    #     print(f"Tiles with FDA applied: {fda}")
    #     print(f"Tiles without FDA applied: {no_fda}")
 
    def create_validation_set(self, split):
        print("Creating validation set...")
        self.validation_dataset_dir = self.feature_tile_dir[0:-10]
        self.validation_dataset_dir = self.validation_dataset_dir + "/validation"

        num_tiles = round(split * len(os.listdir(self.feature_tile_dir)))

        # TODO: Need to shuffle these tiles
        val_feature_tiles = os.listdir(self.feature_tile_dir)[-num_tiles:]
        val_label_tiles = os.listdir(self.label_tile_dir)[-num_tiles:]

        val_feature_tile_root = self.validation_dataset_dir + "/features/"
        val_label_tile_root = self.validation_dataset_dir + "/labels/"

        clear_dir(self.validation_dataset_dir+"/features")
        move_dirs(val_feature_tiles, self.feature_tile_dir, self.validation_dataset_dir+"/features")
        clear_dir(self.validation_dataset_dir+"/labels")
        move_dirs(val_label_tiles, self.label_tile_dir, self.validation_dataset_dir+"/labels")

        self.validation_dataset = SatelliteData(
            self.feature_dir, 
            self.label_dir, 
            self.validation_dataset_dir+"/features",
            self.validation_dataset_dir+"/labels",
            validation_dataset=True,
            random_sample=False            
            )
        
        for i in range(num_tiles): 
            self.validation_dataset.feature_tiles, self.validation_dataset.label_tiles = transform_tif(
                val_feature_tile_root + val_feature_tiles[i], 
                val_label_tile_root + val_label_tiles[i], 
                self.prob, 
                fda=False)
        
        return self.validation_dataset
        

   