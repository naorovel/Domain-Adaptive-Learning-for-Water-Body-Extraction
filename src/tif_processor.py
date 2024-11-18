import os  # If directory manipulation is involved (e.g., path joining, existence checks)
import torch  # For tensor operations
import numpy as np  # For numerical operations like stacking arrays
from torch.utils.data import Dataset  # Base class for custom PyTorch datasets
import rasterio  # For reading .tif files
from rasterio.merge import merge
import glob


class SatelliteDataset(Dataset):
    ##############################
    #This Class is Not tested yet#
    ##############################
    """
    Dataset class. Convert tif files (tiles) into a dataset for model training/testing.
    """
    def __init__(self, feature_dir, label_dir, weight_dir, tiles, mu=None, sigma=None, sample=None):
        # Read Feature Tiles
        feature_files = [f"{feature_dir}/tile_{str(x).rjust(4, '0')}.tif" for x in tiles]
        feature_files = [feature_files[i] for i in sample] if sample is not None else feature_files
        feature_tiles = read_files(feature_files)

        # Normalize Features
        self.mu = torch.mean(feature_tiles, (0, 2, 3)).reshape((1,-1,1,1)) if mu is None else mu
        self.sigma = torch.std(feature_tiles, (0, 2, 3)).reshape((1,-1,1,1)) if sigma is None else sigma
        self.features = normalize(feature_tiles, self.mu, self.sigma)

        # Read Mask Tiles
        mask_files = [f"{label_dir}/tile_{str(x).rjust(4, '0')}.tif" for x in tiles]
        mask_files = [mask_files[i] for i in sample] if sample is not None else mask_files
        self.masks = read_files(mask_files)

        # Read Weight Tiles
        if weight_dir is None:
            self.weights = torch.ones_like(self.masks)
        else:
            weight_files = [f"{weight_dir}/tile_{str(x).rjust(4, '0')}.tif" for x in tiles]
            weight_files = [weight_files[i] for i in sample] if sample is not None else weight_files
            self.weights = read_files(weight_files)

        # Confirm that Number of Features Equals Number of Masks
        assert len(feature_files) == len(mask_files), "Error: Number of features does not match number of masks!"

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return (self.features[idx,:,:,:], self.masks[idx,:,:,:], self.weights[idx,:,:,:])


def read_and_split_tif(file_path, output_dir, tile_size):
    """
    Split a tif into tiles.

    Parameters:
    - file_path: str, path to a tif file.
    - output_file: str, path to a tiles folder.
    - tile_size: int, height=width of a tile.

    Returns:
    - None. Check output_file path for output.
    """
    data, transform, crs = read_tif_file(file_path)
    # Get properties of the data
    _, height, width = data.shape
    # Prepare to split into tiles
    
    tile_count = 0
    for i in range(0, height, tile_size):
        for j in range(0, width, tile_size):
            # Compute the size of the current tile
            tile_height = min(tile_size, height - i)
            tile_width = min(tile_size, width - j)
            # Extract a tile
            tile = data[:, i:i + tile_height, j:j + tile_width]
            # Compute the transform for this tile
            tile_transform = rasterio.transform.from_origin(
                transform.c + j * transform.a,
                transform.f + i * transform.e,
                transform.a,
                transform.e
            )
            # Save the tile
            tile_filename = os.path.join(output_dir, f"tile_{tile_count:04d}.tif")
            with rasterio.open(
                tile_filename,
                'w',
                driver='GTiff',
                height=tile_height,
                width=tile_width,
                count=tile.shape[0],
                dtype=tile.dtype,
                crs=crs,
                transform=tile_transform
            ) as dst:
                dst.write(tile)
            tile_count += 1


def merge_tiles_to_tif(tile_folder, output_file, original_width, original_height, tile_size):
    """
    Merge tiles back to tif image.

    Parameters:
    - tile_folder: str, path to a folder of tiles.
    - output_file: str, path to the output tif file.
    - original_width: int, width of original tif.
    - original_height: int, height of original tif.
    - tile_size: int, height=width of a tile.

    Returns:
    - None. Check output_file path for output.
    """
    # Collect all tile files and sort them by name
    tile_files = sorted(glob.glob(os.path.join(tile_folder, "*.tif")))

    # Open the first tile to get metadata
    with rasterio.open(tile_files[0]) as src:
        num_bands = src.count
        dtype = src.dtypes[0]
        transform = src.transform  # Transform of the first tile

    # Create an empty array for the merged raster
    merged_array = np.zeros((num_bands, original_height, original_width), dtype=dtype)

    # Place tiles into the merged raster
    for idx, tile_file in enumerate(tile_files):
        with rasterio.open(tile_file) as src:
            tile_data = src.read()
            tile_height, tile_width = tile_data.shape[1], tile_data.shape[2]

            # Calculate row and column indices based on the file order
            row = idx // ((original_width + tile_size - 1) // tile_size)
            col = idx % ((original_width + tile_size - 1) // tile_size)

            # Calculate the start and end positions for the tile
            y_start = row * tile_size
            y_end = y_start + tile_height
            x_start = col * tile_size
            x_end = x_start + tile_width

            # Insert the tile into the correct position
            merged_array[:, y_start:y_end, x_start:x_end] = tile_data

    # Compute the global transform for the merged raster
    global_transform = rasterio.transform.from_origin(
        transform.c,  # Top-left X coordinate of the raster
        transform.f,  # Top-left Y coordinate of the raster
        transform.a,  # Pixel width
        transform.e   # Pixel height (usually negative)
    )

    # Write the merged raster to a GeoTIFF file
    with rasterio.open(
        output_file,
        "w",
        driver="GTiff",
        height=original_height,
        width=original_width,
        count=num_bands,
        dtype=dtype,
        crs=None,  # No CRS
        transform=global_transform
    ) as dst:
        dst.write(merged_array)

    print(f"Merged raster saved to {output_file}")

'''
----------------
Helper Functions
----------------
'''
def normalize(x, mu=None, sigma=None):
    return (x - mu) / sigma

def read_files(files):
    tiles = np.stack([rasterio.open(file).read() for file in files])
    return torch.tensor(tiles, dtype=torch.float32)

def read_tif_file(file):
    with rasterio.open(file) as src:
        data = src.read()
        transform = src.transform
        crs = src.crs
    return data, transform, crs

def compare_tif_images(tif1, tif2):
    """
    Compares two GeoTIFF images to check if they have the same pixel values.

    Parameters:
    - tif1: str, path to the first GeoTIFF file.
    - tif2: str, path to the second GeoTIFF file.

    Returns:
    - bool: True if the pixel values are the same, False otherwise.
    - str: Message indicating the result of the comparison.
    """
    try:
        # Open the first image
        with rasterio.open(tif1) as src1:
            data1 = src1.read()
            width1, height1 = src1.width, src1.height
            count1 = src1.count

        # Open the second image
        with rasterio.open(tif2) as src2:
            data2 = src2.read()
            width2, height2 = src2.width, src2.height
            count2 = src2.count

        # Check dimensions
        if (width1, height1, count1) != (width2, height2, count2):
            return False, f"Dimension mismatch: ({width1}, {height1}, {count1}) vs ({width2}, {height2}, {count2})"

        # Compare pixel values
        if not np.array_equal(data1, data2):
            return False, "Pixel values are different between the two images."

        # If everything matches
        return True, "The two images are identical in pixel values."

    except Exception as e:
        return False, f"Error during comparison: {e}"


'''
-------------
Testing Field
-------------
'''
# Training dataset
feature_dir_train = "../data/CN/feature.tif"
label_dir_train = "../data/CN/label.tif"
feature_tiles_train = "../data/CN/tiles/features"
label_tiles_train = "../data/CN/tiles/labels"
feature_tiles_mergeback_train = "../data/CN/tiles/merge/merged_feature.tif"
label_tiles_mergeback_train = "../data/CN/tiles/merge/merged_label.tif"
'''
read_and_split_tif(feature_dir_train, feature_tiles_train, 512)
read_and_split_tif(label_dir_train, label_tiles_train, 512)

merge_tiles_to_tif(feature_tiles_train, feature_tiles_mergeback_train, 20982, 20982, 512)
merge_tiles_to_tif(label_tiles_train, label_tiles_mergeback_train, 20982, 20982, 512)
'''

# Test dataset
feature_dir_test = "../data/BZ/feature.tif"
label_dir_test = "../data/BZ/label.tif"
feature_tiles_test = "../data/BZ/tiles/features"
label_tiles_test = "../data/BZ/tiles/labels"
feature_tiles_mergeback_test = "../data/BZ/tiles/merge/merged_feature.tif"
label_tiles_mergeback_test = "../data/BZ/tiles/merge/merged_label.tif"
'''
read_and_split_tif(feature_dir_test, feature_tiles_test, 512)
read_and_split_tif(label_dir_test, label_tiles_test, 512)

merge_tiles_to_tif(feature_tiles_test, feature_tiles_mergeback_test, 5120, 5120, 512)
merge_tiles_to_tif(label_tiles_test, label_tiles_mergeback_test, 5120, 5120, 512)
'''
# Check if the merged tif has the same pixel values as the original tif.
file1 = "../data/CN/label.tif"
file2 = "../data/CN/tiles/merge/merged_label.tif"
#are_same, message = compare_tif_images(file1, file2)
#print(message)

dataset = SatelliteDataset(
    feature_dir=feature_tiles_test,
    label_dir=label_tiles_test,
    weight_dir=None,
    tiles=range(0, 100),
    mu=None,
    sigma=None,
    sample=None
)

print("Number of tiles:", len(dataset))
print("Feature tiles shape:", dataset.features.shape)
print("Mask tiles shape:", dataset.masks.shape)

features, masks, _ = dataset[0]
print("Features at index 0:", features)
print("Masks at index 0:", masks)