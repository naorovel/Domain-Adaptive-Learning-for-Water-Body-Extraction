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

class SatelliteDataset(Dataset):
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

        # Confirm that Number of Features Equals Number of Masks
        assert len(feature_files) == len(mask_files), "Error: Number of features does not match number of masks!"

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return (self.features[idx,:,:,:], self.masks[idx,:,:,:])

def read_and_split_tif(file_path, output_dir, tile_size):
    """
    Split a tif into tiles.

    Parameters:
    - file_path: str, path to a tif file.
    - output_dir: str, path to a tiles folder.
    - tile_size: int, height=width of a tile.

    Returns:
    - None. Check output_file path for output.
    """
    print("Reading and splitting into tiles...")
    
    # Read the GeoTIFF
    with rasterio.open(file_path) as src:
        data = src.read()  # Read all bands
        transform = src.transform  # Original transform
        crs = src.crs  # Coordinate reference system
    
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
            tile_transform = Affine(
                transform.a,  # Pixel width
                transform.b,  # Always 0 for north-up images
                transform.c + j * transform.a,  # Top-left x-coordinate of the tile
                transform.d,  # Always 0 for north-up images
                transform.e,  # Pixel height (negative for top-left origin)
                transform.f + i * transform.e   # Top-left y-coordinate of the tile
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
    
    print("Tile splitting Done.")

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
    print("Merging tiles...")

    # Collect all tile files and sort them by name
    tile_files = sorted(glob.glob(os.path.join(tile_folder, "*.tif")))

    # Open the first tile to get metadata
    with rasterio.open(tile_files[0]) as src:
        num_bands = src.count
        dtype = src.dtypes[0]
        tile_transform = src.transform  # Transform of the first tile
        crs = src.crs  # CRS from the tiles

    # Create an empty array for the merged raster
    merged_array = np.zeros((num_bands, original_height, original_width), dtype=dtype)

    # Place tiles into the merged raster
    for idx, tile_file in enumerate(tile_files):
        with rasterio.open(tile_file) as src:
            tile_data = src.read()  # Read the tile data
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
    global_transform = Affine(
        tile_transform.a,  # Pixel width
        tile_transform.b,  # Always 0 for north-up images
        tile_transform.c,  # Top-left x-coordinate of the raster
        tile_transform.d,  # Always 0 for north-up images
        tile_transform.e,  # Pixel height (negative for top-left origin)
        tile_transform.f   # Top-left y-coordinate of the raster
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
        crs=crs,  # Use CRS from tiles
        transform=global_transform
    ) as dst:
        dst.write(merged_array)

    print(f"Merged raster saved to {output_file}")

def trim_tif_based_on_tile_size(tif_path, output_path, tif_height, tif_width, tile_height, tile_width, is_CN_feature):
    """
    Trim a .tif file to dimensions that are divisible by the tile size.

    Parameters:
    - tif_path: str, path to the input .tif file.
    - output_path: str, path to save the trimmed .tif file.
    - tif_height: int, height of the original .tif.
    - tif_width: int, width of the original .tif.
    - tile_height: int, height of the tile.
    - tile_width: int, width of the tile.
    - is_CN_feature: Set to true if trim trainning set feature.
    Returns:
    - None. The trimmed .tif file is saved to the output path.
    """
    # Calculate new dimensions
    trimmed_height = (tif_height // tile_height) * tile_height
    trimmed_width = (tif_width // tile_width) * tile_width

    print(f"Original dimensions: {tif_width}x{tif_height}")
    print(f"Trimmed dimensions: {trimmed_width}x{trimmed_height}")

    # Open the input .tif file
    with rasterio.open(tif_path) as src:

        # Create a window for the trimmed region
        window = Window(0, 0, trimmed_width, trimmed_height)

        # Adjust the metadata for the trimmed image
        trimmed_transform = src.window_transform(window)
        new_meta = src.meta.copy()
        new_meta.update({
            "height": trimmed_height,
            "width": trimmed_width,
            "transform": trimmed_transform,
        })

        # Copy color interpretation tags
        color_interp = src.colorinterp if hasattr(src, 'colorinterp') else None

        # Write the trimmed metadata to the new .tif file
        with rasterio.open(output_path, 'w', **new_meta) as dst:
            # Write only the trimmed dimensions
            for band in range(1, src.count + 1):
                band_data = src.read(band, window=window)
                # Flip vertically if the image is inverted
                if is_CN_feature > 0:
                    band_data = band_data[::-1, :]
                dst.write(band_data, band)

            # Preserve color interpretation
            if color_interp:
                dst.colorinterp = color_interp

            # Preserve colormap or photometric tags
            if "photometric" in src.tags():
                dst.update_tags(photometric=src.tags()["photometric"])
            if "colormap" in src.tags():
                dst.update_tags(colormap=src.tags()["colormap"])

    print(f"Trimmed .tif saved to {output_path}")

def random_sample_from_tile(tile_path, output_dir, point, subimage_width, subimage_height):
    """
    Cuts a sub-image from a given point and saves it to the output directory.
    
    Args:
        tile_path (str): Path to the input TIFF image.
        output_dir (str): Directory to save the sub-images.
        point (tuple): (x, y) coordinates of the top-left corner of the sub-image.
        subimage_width (int): Width of the sub-image.
        subimage_height (int): Height of the sub-image.
    """
    # Open the TIFF image
    dataset = gdal.Open(tile_path)
    if dataset is None:
        raise FileNotFoundError(f"Could not open file: {tile_path}")
    
    x, y = point
    band = dataset.GetRasterBand(1)  # Assume the first band is used
    subimage = band.ReadAsArray(x, y, subimage_width, subimage_height)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save sub-image to a new TIFF file
    output_path = os.path.join(output_dir, f"subimage_{x}_{y}.tif")
    driver = gdal.GetDriverByName("GTiff")
    out_dataset = driver.Create(output_path, subimage_width, subimage_height, 1, band.DataType)
    out_dataset.GetRasterBand(1).WriteArray(subimage)
    out_dataset.FlushCache()
    out_dataset = None  # Close the file
    dataset = None  # Close the input file

def process_all_tiles(folder_path, output_base_dir, point, subimage_width, subimage_height):
    """
    Processes all TIFF tiles in a folder, calling random_sample_from_tile on each,
    and saves the outputs to directories named after the tile IDs.
    
    Args:
        folder_path (str): Path to the folder containing TIFF tiles.
        output_base_dir (str): Base directory for saving outputs.
        point (tuple): (x, y) coordinates of the top-left corner of the sub-image.
        subimage_width (int): Width of the sub-image.
        subimage_height (int): Height of the sub-image.
    """
    # Regex pattern to extract tile ID from filenames like tile_0000.tif
    tile_pattern = re.compile(r"tile_(\d+)\.tif")
    
    # Iterate over all files in the folder
    for tile_filename in os.listdir(folder_path):
        # Match files with the tile pattern
        match = tile_pattern.match(tile_filename)
        if not match:
            continue
        
        # Extract tile ID
        tile_id = match.group(1)
        
        # Full path to the current tile
        tile_path = os.path.join(folder_path, tile_filename)
        
        # Output directory for the current tile
        output_dir = os.path.join(output_base_dir, tile_id)
        
        # Call random_sample_from_tile for the current tile
        try:
            random_sample_from_tile(tile_path, output_dir, point, subimage_width, subimage_height)
            print(f"Processed tile {tile_filename} and saved to {output_dir}")
        except RasterioIOError as e:
            print(f"Error reading tile {tile_filename}: {e}")
        except Exception as e:
            print(f"Error processing tile {tile_filename}: {e}")

def process_tiles_with_fda(input_path, output_path, style_folder, num_folders, apply_probability=0.75):
    """
    Process .tif tiles from a specified number of folders (each containing 9 .tif tiles),
    applying FDA with a specified probability using a randomly chosen style image from
    a style folder. Tiles not processed with FDA are copied directly. Saves outputs with
    the same names and subfolder structure as inputs.

    Parameters:
    - input_path: str, path to the folder containing subfolders with .tif tiles.
    - output_path: str, path to the folder where output .tif tiles will be saved.
    - style_folder: str, path to the folder containing style images for FDA.
    - num_folders: int, number of subfolders (each containing 9 tiles) to process.
    - apply_probability: float, probability (0-1) of applying FDA to each tile.

    Returns:
    - A tuple (fda_count, no_fda_count) indicating the number of tiles processed with and without FDA.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Get a list of all subfolders in the input_path
    subfolders = sorted([f for f in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, f))])
    if not subfolders:
        raise ValueError("No subfolders found in the specified input folder.")

    # Limit to the specified number of subfolders
    subfolders = subfolders[:num_folders]

    # Get a list of style images from the style folder
    style_images = [os.path.join(style_folder, f) for f in os.listdir(style_folder) if f.endswith((".png", ".jpg"))]
    if not style_images:
        raise ValueError("No style images found in the specified style folder.")

    fda_count = 0
    no_fda_count = 0

    for subfolder in subfolders:
        subfolder_path = os.path.join(input_path, subfolder)
        output_tile_subfolder = os.path.join(output_path, subfolder)

        # Ensure the output subfolder exists
        if not os.path.exists(output_tile_subfolder):
            os.makedirs(output_tile_subfolder)

        for filename in os.listdir(subfolder_path):
            if filename.endswith(".tif"):
                source_tile_path = os.path.join(subfolder_path, filename)
                output_tile_path = os.path.join(output_tile_subfolder, filename)

                if random.random() < apply_probability:  # Apply FDA with the given probability
                    # Randomly select a style image
                    random_style_image = random.choice(style_images)
                    apply_fda_and_save(source_tile_path, random_style_image, output_tile_path)
                    fda_count += 1
                else:  # Directly copy the file
                    shutil.copy(source_tile_path, output_tile_path)
                    no_fda_count += 1

    return fda_count, no_fda_count


'''
----------------
Helper Functions
----------------
'''

def get_tif_size(tif_path):
    """
    Get the width and height of a TIFF image.
    Args:
        tif_path (str): Path to the TIFF file. 
    Returns:
        tuple: (width, height) of the image in pixels.
    """
    with rasterio.open(tif_path) as dataset:
        width = dataset.width
        height = dataset.height
    return width, height

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
    Compares a subset GeoTIFF image with the corresponding part of a larger GeoTIFF.

    Parameters:
    - tif1: str, path to the first GeoTIFF file (can be the larger image).
    - tif2: str, path to the second GeoTIFF file (can be the subset image).

    Returns:
    - bool: True if the subset matches the corresponding part of the larger image, False otherwise.
    - str: Message indicating the result of the comparison.
    """
    try:
        # Open both images
        with rasterio.open(tif1) as src1, rasterio.open(tif2) as src2:
            # Read metadata and dimensions
            width1, height1 = src1.width, src1.height
            width2, height2 = src2.width, src2.height

            # Check if the subset can fit within the larger image
            if width2 > width1 or height2 > height1:
                return False, "The second image is larger than the first, not a subset."

            # Get geotransforms
            transform1 = src1.transform
            transform2 = src2.transform

            # Calculate pixel coordinates of the subset in the larger image
            col_offset = int((transform2.c - transform1.c) / transform1.a)
            row_offset = int((transform2.f - transform1.f) / transform1.e)

            if col_offset < 0 or row_offset < 0:
                return False, "The subset image does not align with the larger image."

            # Check if the subset goes out of bounds
            if row_offset + height2 > height1 or col_offset + width2 > width1:
                return False, "The subset image extends beyond the bounds of the larger image."

            # Read the subset region from the larger image
            window = rasterio.windows.Window(col_offset, row_offset, width2, height2)
            data1_subset = src1.read(window=window)

            # Read the full data of the subset image
            data2 = src2.read()

            # Compare the pixel values
            if not np.array_equal(data1_subset, data2):
                return False, "The subset does not match the corresponding region in the larger image."

            # If everything matches
            return True, "The subset matches the corresponding region in the larger image."

    except Exception as e:
        return False, f"Error during comparison: {e}"

def check_subimage_matches_original(original_path, subimage_path, point, subimage_width, subimage_height):
    """
    Checks if the pixels in the sub-image match the original image pixels in the corresponding range.
    
    Args:
        original_path (str): Path to the original TIFF image.
        subimage_path (str): Path to the saved sub-image TIFF.
        point (tuple): (x, y) coordinates of the top-left corner of the sub-image in the original image.
        subimage_width (int): Width of the sub-image.
        subimage_height (int): Height of the sub-image.
    
    Returns:
        bool: True if the sub-image matches the original image, False otherwise.
    """
    x, y = point
    
    # Open the original image
    with rasterio.open(original_path) as original:
        window = rasterio.windows.Window(x, y, subimage_width, subimage_height)
        original_pixels = original.read(window=window)
    
    # Open the sub-image
    with rasterio.open(subimage_path) as subimage:
        subimage_pixels = subimage.read()
    
    # Check if the shapes match
    if original_pixels.shape != subimage_pixels.shape:
        print(f"Shape mismatch: Original {original_pixels.shape}, Sub-image {subimage_pixels.shape}")
        return False
    
    # Check if the pixel values match
    if not np.array_equal(original_pixels, subimage_pixels):
        print("Pixel values do not match.")
        return False
    
    return True

def fix_geotransform(input_tif, output_tif):
    """
    Fix the geotransform to set a "bottom-up" raster orientation.
    Parameters:
    - input_tif: str, path to the input GeoTIFF file with incorrect geotransform.
    - output_tif: str, path to save the corrected GeoTIFF.
    """
    with rasterio.open(input_tif) as src:
        # Calculate the correct geotransform for a bottom-up raster
        transform = from_origin(0.0, 0.0, 1.0, 1.0)  # origin (0,0), resolution (1,1)
        
        # Update metadata
        new_meta = src.meta.copy()
        new_meta.update({"transform": transform})

        # Save the corrected raster
        with rasterio.open(output_tif, "w", **new_meta) as dst:
            dst.write(src.read())

def show_geotransform(tif_path):
    """
    Show geotransform metadata of a GeoTIFF file.
    """
    with rasterio.open(tif_path) as src:
        print(f"File: {tif_path}")
        print(f"Geotransform: {src.transform}")

def sample_random_points(image_width, image_height, subimage_width, subimage_height, num_samples=9, rng=None):
    """
    Randomly samples points within valid ranges for sub-image extraction using a random number generator instance.
    
    Args:
        image_width (int): Width of the original image.
        image_height (int): Height of the original image.
        subimage_width (int): Width of the sub-image.
        subimage_height (int): Height of the sub-image.
        num_samples (int): Number of points to sample.
        rng (random.Random, optional): A random number generator instance.
    
    Returns:
        list of tuples: List of (x, y) coordinate points.
    """
    if rng is None:
        rng = random
    
    max_x = image_width - subimage_width
    max_y = image_height - subimage_height
    points = [(rng.randint(0, max_x), rng.randint(0, max_y)) for _ in range(num_samples)]
    return points

def random_samples_from_tile(tile_path, output_dir, points, subimage_width, subimage_height):
    """
    Cuts sub-images from a given list of points and saves them to the output directory using rasterio.
    
    Args:
        tile_path (str): Path to the input TIFF image.
        output_dir (str): Directory to save the sub-images.
        points (list of tuples): List of (x, y) coordinates of the top-left corners of the sub-images.
        subimage_width (int): Width of the sub-images.
        subimage_height (int): Height of the sub-images.
    """
    print("Applying FDA...")
    with rasterio.open(tile_path) as dataset:
        a = basename(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        for b, point in enumerate(points, start=1):
            x, y = point
            
            if x + subimage_width > dataset.width or y + subimage_height > dataset.height:
                print(f"Skipping point ({x}, {y}) - subimage out of bounds.")
                continue
            
            # Define the window for the sub-image
            window = Window(x, y, subimage_width, subimage_height)
            
            try:
                # Read all bands using the window
                subimage = dataset.read(window=window)  # Reads all bands
            except ValueError:
                print(f"Skipping point ({x}, {y}) - error reading subimage.")
                continue
            
            # Define output path
            output_path = os.path.join(output_dir, f"sample_{a}_{b}.tif")
            
            # Preserve nodata value
            nodata_value = dataset.nodata
            
            # Save the sub-image
            with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=subimage.shape[1],
                width=subimage.shape[2],
                count=dataset.count,  # Number of bands
                dtype=subimage.dtype,
                crs=dataset.crs,
                transform=rasterio.windows.transform(window, dataset.transform),
                nodata=nodata_value,  # Preserve NoData value
            ) as out_dataset:
                out_dataset.write(subimage)
    print("FDA Done.")
            
def random_samples_from_tiles(feature_dir, label_dir, output_base_dir, subimage_size, num_samples=9):
    """
    Processes corresponding feature and label tiles, generates random samples for each,
    and saves them in separate output directories.

    Args:
        feature_dir (str): Path to the folder containing feature tile TIFF files.
        label_dir (str): Path to the folder containing label tile TIFF files.
        output_base_dir (str): Base directory to save the sub-image samples.
        subimage_size (int): Size (width and height) of the sub-images.
        num_samples (int): Number of random samples to generate per tile.
    """
    print("Random sampling from both feature and label tiles...")
    # Initialize random sample generator
    random_sample_generator = random.Random(42)

    # Ensure output directories exist
    output_features_dir = os.path.join(output_base_dir, "features")
    output_labels_dir = os.path.join(output_base_dir, "labels")
    os.makedirs(output_features_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    # Process each feature tile
    for tile_name in os.listdir(feature_dir):
        if tile_name.endswith(".tif"):  # Ensure it's a TIFF file
            feature_tile_path = os.path.join(feature_dir, tile_name)
            label_tile_path = os.path.join(label_dir, tile_name)

            # Check if corresponding label tile exists
            if not os.path.exists(label_tile_path):
                print(f"Skipping {tile_name} - corresponding label tile not found.")
                continue

            # Extract tile ID from the file name
            tile_id = os.path.splitext(tile_name)[0].split('_')[-1]

            # Define output directories for this tile
            output_feature_dir = os.path.join(output_features_dir, tile_id)
            output_label_dir = os.path.join(output_labels_dir, tile_id)
            os.makedirs(output_feature_dir, exist_ok=True)
            os.makedirs(output_label_dir, exist_ok=True)

            # Assuming tiles are 512x512
            random_points = sample_random_points(512, 512, subimage_size, subimage_size,
                                                 num_samples=num_samples, rng=random_sample_generator)

            # Generate random samples for the feature tile
            random_samples_from_tile(feature_tile_path, output_feature_dir, random_points, subimage_size, subimage_size)

            # Generate random samples for the label tile
            random_samples_from_tile(label_tile_path, output_label_dir, random_points, subimage_size, subimage_size)
    print("Random sampling Done.")

'''
-------------
Testing Field
-------------
'''
# Training dataset
feature_dir_train = "../data/CN/feature_trimmed.tif"
label_dir_train = "../data/CN/label_trimmed.tif"
feature_tiles_train = "../data/CN/tiles/features"
label_tiles_train = "../data/CN/tiles/labels"
feature_tiles_mergeback_train = "../data/CN/tiles/merge/merged_feature.tif"
label_tiles_mergeback_train = "../data/CN/tiles/merge/merged_label.tif"

# Test dataset
feature_dir_test = "../data/BZ/feature.tif"
label_dir_test = "../data/BZ/label.tif"
feature_tiles_test = "../data/BZ/tiles/features"
label_tiles_test = "../data/BZ/tiles/labels"
feature_tiles_mergeback_test = "../data/BZ/tiles/merge/merged_feature.tif"
label_tiles_mergeback_test = "../data/BZ/tiles/merge/merged_label.tif"



# splite and merge back -- training 
#read_and_split_tif(label_dir_train, label_tiles_train, 512)
'''
read_and_split_tif(feature_dir_train, feature_tiles_train, 512)
read_and_split_tif(label_dir_train, label_tiles_train, 512)

merge_tiles_to_tif(feature_tiles_train, feature_tiles_mergeback_train, 20480, 20480, 512)
merge_tiles_to_tif(label_tiles_train, label_tiles_mergeback_train, 20480, 20480, 512)
'''
# splite and merge back -- test
'''
read_and_split_tif(feature_dir_test, feature_tiles_test, 512)
read_and_split_tif(label_dir_test, label_tiles_test, 512)

merge_tiles_to_tif(feature_tiles_test, feature_tiles_mergeback_test, 5120, 5120, 512)
merge_tiles_to_tif(label_tiles_test, label_tiles_mergeback_test, 5120, 5120, 512)
'''

# compare if 2 images are the same
'''
file1 = "../data/CN/feature.tif"
file2 = "../data/CN/corrected_geotransform.tif"
are_same, message = compare_tif_images(feature_dir_train, feature_tiles_mergeback_train)
print(message)

'''

# create database
'''
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
'''

# random sampling from a sigle tile (test purpose)
'''
random_sample_generator = random.Random(42)
points_test = sample_random_points(512, 512, 256, 256, num_samples=9, rng=random_sample_generator)

#print("Points from first call:", points_test)
tile_path1 = "../data/CN/tiles/features/tile_0020.tif"
output_dir1 = "../data/CN/random_samples/0020"
subimage_size=256
random_samples_from_tile(tile_path1, output_dir1, points_test, subimage_size, subimage_size)
'''

# random sampling from all tiles
#random_samples_from_tiles(feature_tiles_train, label_tiles_train,"../data/CN/random_samples", 256, num_samples=9)

# Test fda
'''
source = "../data/CN/tiles/features/tile_0000.tif"
target = "../data/Style/000012.png"
output = "../data/CN/processed/result.tif"

apply_fda_and_save(source, target, output)
'''
source = "../data/CN/random_samples/features"
target = "../data/Style"
output = "../data/CN/processed"
fda, no_fda = process_tiles_with_fda(source, output, target, 10, apply_probability=0.75)
print(f"Tiles with FDA applied: {fda}")
print(f"Tiles without FDA applied: {no_fda}")
