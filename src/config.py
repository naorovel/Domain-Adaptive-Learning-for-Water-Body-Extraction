#########################################
# Global Variables
#########################################

# Needed for dataset generation:
training_dirs = {
    'feature_dir': "../data/CN/feature_trimmed.tif",
    'label_dir': "../data/CN/label_trimmed.tif",

    'feature_tiles': "../data/CN/tiles/features",
    'label_tiles': "../data/CN/tiles/labels",

    'feature_mergeback': "../data/CN/tiles/merge/merged_feature.tif",
    'label_mergeback': "../data/CN/tiles/merge/merged_label.tif",

    'split': "../data/CN/random_samples",
}

validation_dirs = {
    'feature_tiles': "../data/CN/tiles/validation/features",
    'label_tiles': "../data/CN/tiles/validation/labels",
}

# Needed for test dataset generation
test_dirs = {
    'feature_dir': "../data/BZ/feature.tif",
    'label_dir': "../data/BZ/label.tif",

    'feature_tiles': "../data/BZ/tiles/features",
    'label_tiles': "../data/BZ/tiles/labels",

    'feature_mergeback': "../data/BZ/tiles/merge/merged_feature.tif",
    'label_mergeback': "../data/BZ/tiles/merge/merged_label.tif",
}

# For saving processed data
"""
Processed data dir: 
- train (subdirs contain patches)
    - baseline_transforms (includes basic transformations only)
        - features
        - labels
    - water_only (includes all transforms, FDA is only on water-containing tiles)
        - features
        - labels
    - all (includes all transformations, FDA is on selection of all tiles)
        - features
        - labels
- validation (subdirs contain tiles)
    - features
    - labels
- test (subdirs contain tiles)
    - features
    - labels
"""
processed_dirs = {
    'train': "../data/CN/processed/train",
    'val': "../data/CN/processed/validation",
    'test': "../data/BZ/processed/test",
} 
 
style_dirs = {
    'style_dir_water': '../data/style/tiles/water',
    'style_dir_all': '../data/style/tiles/all',
}

training_prob = {
    'p_FDA': 0.75,
    'p_hflip': 0.5,
    'p_vflip': 0.5,
    'p_90rot': 0.25
}

val_split = 0.1

num_samples = 9
