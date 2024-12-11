from torch.utils.data import Dataset  # Base class for custom PyTorch datasets
from tif_utils import *

class SatelliteData(Dataset): 

    feature_dir = ""
    label_dir = ""
    feature_tile_dir = ""
    label_tile_dir = ""
    split_tile_dir = ""
    processed_dir = ""
    mergeback_tif_dir = ""

    split_size = 0

    def __init__(self, split_size):
        self.split_size = split_size

    def load_dataset(self): 
        print("Loading dataset...")

    def generate_split_tiles(self):
        self.split_tiles()
        self.center_tiles() 

    def split_tiles(self):
        """
        Splits and saves tiles into self.feature_tile_dir and self.label_tile_dir
        """
        clear_dir(self.feature_tile_dir)
        clear_dir(self.label_tile_dir)
        read_and_split_tif(self.feature_dir, self.feature_tile_dir, self.split_size)
        read_and_split_tif(self.label_dir, self.label_tile_dir, self.split_size)
        
        print(f"New tiles of size {self.split_size} were created.")
        self.verify_split()

    
    def center_tiles(self):
        """
        Centers and saves changes to split tiles. 
        """
        print("Centering tiles...")
        # Read all feature tiles in the directory first
        feature_tiles = self.get_tile_paths(self.feature_tile_dir)

        # For each tile, perform centering
        for tile in feature_tiles: 
            center_tile(tile)
            
    def get_tile_paths(self, dir): 
        """
        Gets complete paths for tiles in directory. 
        """
        tiles = os.listdir(dir)
        tiles = [dir + os.path.sep + tile for tile in tiles]
        return tiles

    def verify_split(self): 
        """
        Confirms that the split is non-overlapping and results in the original TIF. 
        """
        print("Verifying split...")
        width, height = get_tif_size(self.feature_dir)
        merge_tiles_to_tif(self.feature_tile_dir, self.mergeback_feature_dir, 
                           width, height, self.split_size)
        merge_tiles_to_tif(self.label_tile_dir, self.mergeback_label_dir, 
                           width, height, self.split_size)
        are_same_feature, message = compare_tif_images(self.feature_dir, self.mergeback_feature_dir)
        are_same_label, message = compare_tif_images(self.label_dir, self.mergeback_label_dir)
        assert are_same_feature, are_same_label

    def set_dirs(self,
                 feature_dir, 
                 label_dir, 
                 feature_tile_dir, 
                 label_tile_dir,  
                 processed_dir, 
                 split_tile_dir = "",
                 mergeback_feature_dir="",
                 mergeback_label_dir = ""): 
        self.feature_dir = feature_dir
        self.label_dir = label_dir
        self.feature_tile_dir = feature_tile_dir
        self.label_tile_dir = label_tile_dir
        self.split_tile_dir = split_tile_dir
        self.processed_dir = processed_dir
        self.mergeback_feature_dir = mergeback_feature_dir
        self.mergeback_label_dir = mergeback_label_dir

    def move_dir(self, in_dir, out_dir): 
        clear_dir(out_dir)

        list_dir = os.listdir(in_dir)

        for sub_dir in list_dir:
            dir_to_move = os.path.join(in_dir, sub_dir)
            shutil.move(dir_to_move, out_dir)