import os, shutil  # If directory manipulation is involved (e.g., path joining, existence checks)
import torch  # For tensor operations
import numpy as np  # For numerical operations like stacking arrays
import rasterio  # For reading .tif files
from rasterio.windows import Window
from rasterio.transform import from_origin
from rasterio.transform import Affine
import random
from os.path import basename
import glob
from FDA.FDA import apply_fda_and_save

random.seed(10)

########################################################################
# A collection of functions to process TIF files.
########################################################################

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

def read_files(dir):
    filenames = filter(lambda k: k[-4:] == ".tif",os.listdir(dir))
    tiles = np.stack([rasterio.open(dir + "/" + file).read() for file in filenames])
    return torch.tensor(tiles, dtype=torch.float32)

def read_tif_file(file):
    with rasterio.open(file) as src:
        data = src.read()
        transform = src.transform
        crs = src.crs,
        num_bands = src.count
        dtype = src.dtypes[0]
    return data, transform, crs, num_bands, dtype

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
    # print("Merging tiles...")

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

    # print(f"Merged raster saved to {output_file}")

def clear_dir(dir): 
    # TODO: Fix bug - currently deletes all files including gitkeep. Should not delete .gitkeep
    for filename in os.listdir(dir): 
        file_path = os.path.join(dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def transform_patch(random_sample_dir, patch_dir, filename, prob, save_dir): 
    # Gets a path to a patch
    feature_dir = random_sample_dir + os.sep + "features" + os.sep + patch_dir
    mask_dir = random_sample_dir + os.sep + "labels" + os.sep + patch_dir

    f_data, l_data = transform_tif(feature_dir, mask_dir, prob, patch_dir, filename, save_dir)

    return f_data, l_data

def transform_tif(feature_dir, mask_dir, prob, patch_dir, filename, save_dir, fda=True): 
    f_data, f_transform, f_crs, num_bands, dtype = read_tif_file(feature_dir)
    l_data, l_transform, l_crs, num_bands, dtype = read_tif_file(mask_dir)
    data = [f_data, l_data]
    __, height, width = f_data.shape

    #f_data = self.apply_fda(water=False) # Do not conduct FDA on mask
    data = apply_flip([f_data, l_data], True, prob["p_hflip"])
    data = apply_flip([f_data, l_data], False, prob["p_vflip"])
    data = apply_rotation([f_data, l_data], 90, prob["p_90rot"])
    #data = apply_rotation([f_data, l_data], 270, prob["p_270rot"])
    f_data = torch.from_numpy(data[0].copy())
    l_data = torch.from_numpy(data[1].copy())

    if save_dir is not None: 
        feat_out_path = save_dir + os.sep + 'features' + filename
        label_out_path = save_dir + os.sep + 'labels' + filename

        with rasterio.open(feat_out_path, 'w', driver='GTiff', height=height,
                           width=width,
                           dtype = data[0].dtype, 
                           #crs = f_crs,
                           count = 4,
                           transform = f_transform) as dst: 
            dst.write(data[0].copy())
        with rasterio.open(label_out_path, 'w', driver='GTiff', height=height,
                           width=width,
                           dtype = data[1].dtype, 
                           #crs = f_crs,
                           count=1,
                           transform = f_transform) as dst: 
            dst.write(data[1].copy())
    
    return f_data, l_data

def apply_flip(data, horizontal, prob):
    res = []
    if horizontal: 
        for i in range(len(data)): 
            if prob_succeed(prob): 
                res.append(np.flip(data[i], axis=2))
            else: 
                res.append(data[i])
    else: 
        for i in range(len(data)): 
            if prob_succeed(prob): 
                res.append(np.flip(data[i], axis=1))
            else: 
                res.append(data[i])
    return res

def apply_rotation(data, deg, prob):
    res = []
    if deg==90: 
        for i in range(len(data)): 
            if prob_succeed(prob): 
                res.append(data[i].swapaxes(1, 2))
            else: 
                res.append(data[i])
    # elif deg==270: # assumed to be 270 degrees: 
    #     for i in range(len(data)): 
    #         if prob_succeed(prob): 
    #             res.append(np.rot90(data[i], k=1, axes=(0,1)))
    #         else: 
    #             res.append(data[i])
    return res

def prob_succeed(p): 
    return random.random() < p

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

def move_dirs(source_dir_list, source_dir, dest):
    for sub_dir in source_dir_list:
        dir_to_move = os.path.join(source_dir, sub_dir)
        shutil.move(dir_to_move, dest)     