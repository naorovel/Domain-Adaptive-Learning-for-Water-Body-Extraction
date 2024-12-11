from satellite_data_parent import SatelliteData
from validation_data import ValidationData
from tif_utils import *
import config
import os


class TrainingData(SatelliteData): 
    
    patch_size = 0
    val_split = 0
    prob = {}
    feature_patch_dir = ""
    label_patch_dir = ""

    def __init__(self, split_size):
        self.patch_size = split_size / 2
        self.val_split = config.val_split
        self.prob = config.training_prob
        self.num_samples = config.num_samples

        self.set_dirs()

        super().__init__(split_size)

        self.load_dataset()

    def load_dataset(self):
        super().load_dataset()
    
    def set_dirs(self): 
        super().set_dirs(
            config.training_dirs['feature_dir'],
            config.training_dirs['label_dir'],
            config.training_dirs['feature_tiles'],
            config.training_dirs['label_tiles'],
            config.processed_dirs['train'],
            config.training_dirs['split'],
            config.training_dirs['feature_mergeback'],
            config.training_dirs['label_mergeback']
        )
        self.feature_patch_dir = os.path.join(config.training_dirs['split'], 'features')
        self.label_patch_dir = os.path.join(config.training_dirs['split'], 'labels')
        self.style_dir_water = config.style_dirs['style_dir_water']
        self.style_dir_all = config.style_dirs['style_dir_all']


    def generate_new(self): 
        super().generate_split_tiles()
        
    def generate_new_validation(self): 
        print("Generating new validation")
        validation = ValidationData(self.split_size, self.val_split, self.feature_tile_dir, self.label_tile_dir)
        return validation

    def generate_new_transformed(self, fda=False, water=True): 
        save_dir = self.get_processed_dir(fda, water)
        self.generate_patches()
        self.process_patches(fda, water)

    def generate_patches(self): 
        print("Generating new patches")
        clear_dir(os.path.join(self.split_tile_dir, 'features'))
        clear_dir(os.path.join(self.split_tile_dir, 'labels'))

        random_samples_from_tiles(
            self.feature_tile_dir, self.label_tile_dir,
            self.split_tile_dir, self.patch_size,
            self.num_samples
        )
        num_patches = sum([len(files) for r, d, files in os.walk(os.path.join(self.split_tile_dir,"features"))])
        print(f"Generated {num_patches} patches.")

    def process_patches(self, fda, water_only):
        print("Processing patches") 
    
        # Clear output directory: 
        clear_dir(os.path.join(self.processed_dir, 'features'))
        clear_dir(os.path.join(self.processed_dir, 'labels'))

        feature_patches = self.get_patch_dirs(self.feature_patch_dir)
        label_patches = self.get_patch_dirs(self.label_patch_dir)

        num_patches = len(feature_patches)
        style_patches = self.get_style_patch_dirs(fda, water_only, num_patches)

        filenames = [patch.split(os.path.sep)[-1] for patch in feature_patches]

        for i in range(num_patches): 
            if fda: 
                transform_patch(feature_patches[i], label_patches[i], self.prob, fda, filenames[i], self.processed_dir, style_patches[i])
            else: 
                transform_patch(feature_patches[i], label_patches[i], self.prob, fda, filenames[i], self.processed_dir)
            
        print(f"Processed {num_patches} patches.")

        self.move_to_processed(fda, water_only)

    def get_patch_dirs(self, dir): 
        """
        Get complete paths for patches in directory
        """
        patch_dirs = []
        for root, dirs, files in os.walk(dir):
            for name in files: 
                dirname = root.split(os.path.sep)[-1]
                patch_dirs.append(os.path.join(dir, dirname, name))

        return patch_dirs
    
    def get_style_patch_dirs(self, fda, water_only, n): 
        """
        Get complete paths for FDA tiles for styling
        """
        if fda and water_only: # Water only
            return self.get_tile_paths(self.style_dir_water)

        elif fda and not water_only: # All tiles
            return self.get_tile_paths(self.style_dir_water) + self.get_tile_paths(self.style_dir_all)
        else: 
            return []

    def move_to_processed(self, fda, water_only): 
        """
        Move processed patches from temp to dedicated processed output directory.
        """
        output_dir = self.get_processed_dir(fda, water_only)

        clear_dir(os.path.join(output_dir, 'features'))
        clear_dir(os.path.join(output_dir, 'labels'))

        self.move_dir(os.path.join(self.processed_dir, 'features'), os.path.join(output_dir, 'features'))
        self.move_dir(os.path.join(self.processed_dir, 'labels'), os.path.join(output_dir, 'labels'))

    def get_processed_dir(self, fda, water):
        """
        Get appropriate output directory given input params.
        """
        if not fda and not water: 
            return self.get_processed_baseline_dir()
        elif fda and not water: 
            return self.get_processed_water_dir()
        else:
            return self.get_processed_all_dir()

    def get_processed_baseline_dir(self): 
        return os.path.join(self.processed_dir, 'baseline')
    
    def get_processed_water_dir(self): 
        return os.path.join(self.processed_dir, 'water_only')
    
    def get_processed_all_dir(self): 
        return os.path.join(self.processed_dir, 'all')