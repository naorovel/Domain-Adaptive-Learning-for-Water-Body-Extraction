from satellite_data_parent import SatelliteData
import config
import math
import random
import shutil
from tif_utils import *

random.seed(10)

class ValidationData(SatelliteData): 

    def __init__(self, split_size, val_split, feature_tiles, label_tiles):
        self.set_dirs()
        self.val_split = val_split
        super().__init__(split_size)

        self.load_dataset()

        self.generate_new(feature_tiles, label_tiles)

    def load_dataset(self):
        super().load_dataset()
    
    def set_dirs(self): 
        super().set_dirs(
            config.training_dirs['feature_dir'],
            config.training_dirs['label_dir'],
            config.validation_dirs['feature_tiles'],
            config.validation_dirs['label_tiles'],
            config.processed_dirs['val'],
        )

    def generate_new(self, feature_tiles, label_tiles): 
        print("Generating validation set...")
        # Get paths for tiles in feature_tiles and label_tiles directories
        feature_tiles = self.get_tile_paths(feature_tiles)
        label_tiles = self.get_tile_paths(label_tiles)

        # Get number of tiles needed, given self.val_split, randomly select n tiles.  
        total_tiles = len(feature_tiles)
        picked = random.sample(range(total_tiles), self.get_num_tiles(total_tiles))

        # Move selected tiles to the validation directory
        clear_dir(self.feature_tile_dir)
        clear_dir(self.label_tile_dir)
        for i in picked: 
            shutil.move(feature_tiles[i], self.feature_tile_dir)
            shutil.move(label_tiles[i], self.label_tile_dir)
        
        print(f"{len(picked)} tiles were moved to the validation set.")
        self.move_to_processed()
        
    def get_num_tiles(self, n): 
        return math.ceil(n*self.val_split)
    
    def move_to_processed(self): 
        super().move_dir(self.feature_tile_dir, os.path.join(self.processed_dir, 'features'))
        super().move_dir(self.label_tile_dir, os.path.join(self.processed_dir, 'labels'))
        