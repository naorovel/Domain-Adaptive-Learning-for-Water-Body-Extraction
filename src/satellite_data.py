import torch  # For tensor operations
from torch.utils.data import Dataset  # Base class for custom PyTorch datasets
from tif_utils import *

class SatelliteData(Dataset):
    """
    Dataset class. Convert tif files (tiles) into a dataset for model training/testing.
    """
    
    mu: torch.float16
    sigma: torch.float16
    feature_tiles: torch.tensor = []
    label_tiles:torch.tensor = []

    def __init__(self, feature_dir, label_dir,
                 feature_tile_dir, label_tile_dir,
                 feature_tiles_mergeback_dir=None, 
                 label_tiles_mergeback_dir=None,
                 sample=None,
                 split_size = 512,
                 p_FDA = 0.75,
                 p_hflip = 0.5,
                 p_vflip = 0.5,
                 p_90rot = 0.25,
                 p_180rot = 0.25):
        self.feature_dir = feature_dir
        self.label_dir = label_dir
        self.feature_tile_dir = feature_tile_dir
        self.label_tile_dir = label_tile_dir
        self.feature_tiles_mergeback_dir = feature_tiles_mergeback_dir
        self.label_tiles_mergeback_dir = label_tiles_mergeback_dir
        self.sample = sample
        self.split_size = split_size

        self.generate_split_tiles()
        self.read_split_tiles()
        self.feature_tiles = self.normalize_tiles(self.feature_tiles)


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

    def random_subset(self, prob): 
        """
        Generates a random subset of tiles names given a probability of selection.
        prob: float16 in (0, 1). 

        Returns: A set of tile names selected with probability "prob".
        """
        # TODO: Implementation

    def apply_transform(self, transform, prob):
        """
        Applies the function 'transform' (tile -> tile) on each tile generated
        in a random subset with probability 'prob'. 

        Returns: A set of transformed TIFs. 
        """
        # TODO: Implementation

    