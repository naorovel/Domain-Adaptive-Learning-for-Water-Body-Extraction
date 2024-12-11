from satellite_data_parent import SatelliteData
import config
import os

class TestData(SatelliteData): 
    
    def __init__(self, split_size):
        self.set_dirs()

        super().__init__(split_size)

        self.load_dataset()

    def load_dataset(self):
        super().load_dataset()
    
    def set_dirs(self): 
        super().set_dirs(
            config.test_dirs['feature_dir'],
            config.test_dirs['label_dir'],
            config.test_dirs['feature_tiles'],
            config.test_dirs['label_tiles'],
            config.processed_dirs['test'],
            mergeback_feature_dir=config.test_dirs['feature_mergeback'],
            mergeback_label_dir=config.test_dirs['label_mergeback']
        )

    def generate_new(self): 
        super().generate_split_tiles()
        self.move_to_processed()

    def move_to_processed(self): 
        super().move_dir(self.feature_tile_dir, os.path.join(self.processed_dir, 'features'))
        super().move_dir(self.label_tile_dir, os.path.join(self.processed_dir, 'labels'))