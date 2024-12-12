import numpy as np
from PIL import Image
from .utils import FDA_source_to_target_np
import rasterio
from scipy.ndimage import map_coordinates


def apply_fda_and_save(source_path, target_path, output_path):
    """
    Apply Fourier Domain Adaptation (FDA) to the first three bands (RGB) of a source GeoTIFF
    and combine with the unchanged 4th band to save as a new GeoTIFF.

    Parameters:
    - source_path: str, path to the source GeoTIFF file.
    - target_path: str, path to the target image file (e.g., .png or .jpg).
    - output_path: str, path to save the resulting GeoTIFF file.

    Returns:
    - None. Saves the output GeoTIFF at the specified path.
    """

    # Read the GeoTIFF file
    with rasterio.open(source_path) as src:
        # Read the source metadata
        src_meta = src.meta.copy()
        
        if src.count >= 4:  # Ensure at least 4 bands (RGB + Extra)
            im_src_rgb = src.read([1, 2, 3])  # Read the first 3 bands (RGB)
            im_src_extra = src.read(4)  # Read the 4th band
        else:
            raise ValueError("The source image must have at least 4 bands.")

        # Convert RGB bands to (height, width, channels)
        im_src_rgb = np.transpose(im_src_rgb, (1, 2, 0))  # (bands, height, width) -> (height, width, channels)

    # Normalize the RGB bands to [0, 255]
    im_src_rgb = im_src_rgb.astype(np.float32)
    im_src_rgb = (im_src_rgb - im_src_rgb.min()) / (im_src_rgb.max() - im_src_rgb.min()) * 255

    # Read and resize the target image to match the source dimensions
    im_trg = Image.open(target_path).convert('RGB')
    im_trg = im_trg.resize((im_src_rgb.shape[1], im_src_rgb.shape[0]), Image.BICUBIC)
    im_trg = np.asarray(im_trg, np.float32)

    # Transpose for FDA
    im_src_rgb = im_src_rgb.transpose((2, 0, 1))  # (height, width, channels) -> (channels, height, width)
    im_trg = im_trg.transpose((2, 0, 1))

    # Apply FDA to the RGB bands
    fda_result = FDA_source_to_target_np(im_src_rgb, im_trg, L=0.01)

    # Convert FDA result back to (height, width, channels)
    fda_result = fda_result.transpose((1, 2, 0))

    # Ensure FDA result is in the [0, 255] range
    fda_result = np.clip(fda_result, 0, 255).astype(np.uint8)

    # Keep the 4th band unchanged
    im_src_extra = np.clip(im_src_extra, 0, 255).astype(np.uint8)

    # Combine the FDA-processed RGB bands with the unchanged 4th band
    output_array = np.dstack((fda_result, im_src_extra))

    # Prepare metadata for output GeoTIFF
    src_meta.update({
        "driver": "GTiff",
        "dtype": "uint8",
        "count": 4,  # Output will have 4 bands (3 RGB + 1 extra)
    })

    # Save the result as a GeoTIFF with the same geometry and coordinates as the source
    with rasterio.open(output_path, "w", **src_meta) as dst:
        # Write each band of the processed image
        for i in range(3):  # Write RGB bands
            dst.write(output_array[..., i], i + 1)
        dst.write(im_src_extra, 4)  # Write the unchanged 4th band

    #print(f"Result saved to {output_path}")

def apply_fda(source_data, target_path):
    """
    Apply Fourier Domain Adaptation (FDA) to the first three bands (RGB) of a source GeoTIFF
    and combine with the unchanged 4th band to save as a new GeoTIFF.

    Parameters:
    - source_data: 2-element list representing data from original TIF files. First element is the 4-Band raster, and the 2nd is the mask. FDA is not performed on the mask. 
    - target_path: str, path to the target image file (e.g., TIF).

    Returns:
    - None. Saves the output GeoTIFF at the specified path.
    """

    f_data = source_data[0]
    l_data = source_data[1]

    # Read the GeoTIFF file
    with rasterio.open(target_path) as src:
        # Read the source metadata
        src_meta = src.meta.copy()
        
        if src.count == 4:  # Ensure at least 4 bands (RGB + Extra)
            im_target_rgb = src.read([1, 2, 3])  # Read the first 3 bands (RGB)
            im_target_extra = src.read(4)  # Read the 4th band
        else:
            raise ValueError("The target image must have at least 4 bands.")

        # Convert RGB bands to (height, width, channels)
        im_target_rgb = np.transpose(im_target_rgb, (1, 2, 0))  # (bands, height, width) -> (height, width, channels)

    # Normalize the RGB bands to [0, 255]
    im_target_rgb = im_target_rgb.astype(np.float32)
    im_target_rgb = (im_target_rgb - im_target_rgb.min()) / (im_target_rgb.max() - im_target_rgb.min()) * 255

    # Read and resize the target image to match the source dimensions
    im_trg = Image.open(target_path).convert('RGB')
    im_trg = im_trg.resize((im_target_rgb.shape[1], im_target_rgb.shape[0]), Image.BICUBIC)
    im_trg = np.asarray(im_trg, np.float32)

    # Transpose for FDA
    im_target_rgb = im_target_rgb.transpose((2, 0, 1))  # (height, width, channels) -> (channels, height, width)
    im_trg = im_trg.transpose((2, 0, 1))

    # Apply FDA to the RGB bands
    fda_result = FDA_source_to_target_np(im_target_rgb, im_trg, L=0.01)

    # Convert FDA result back to (height, width, channels)
    fda_result = fda_result.transpose((1, 2, 0))

    # Ensure FDA result is in the [0, 255] range
    fda_result = np.clip(fda_result, 0, 255).astype(np.uint8)

    # Keep the 4th band unchanged
    im_src_extra = np.clip(im_src_extra, 0, 255).astype(np.uint8)

    # Combine the FDA-processed RGB bands with the unchanged 4th band
    output_array = np.dstack((fda_result, im_src_extra))

    # Prepare metadata for output GeoTIFF
    src_meta.update({
        "driver": "GTiff",
        "dtype": "uint8",
        "count": 4,  # Output will have 4 bands (3 RGB + 1 extra)
    })

    # Save the result as a GeoTIFF with the same geometry and coordinates as the source
    with rasterio.open(output_path, "w", **src_meta) as dst:
        # Write each band of the processed image
        for i in range(3):  # Write RGB bands
            dst.write(output_array[..., i], i + 1)
        dst.write(im_src_extra, 4)  # Write the unchanged 4th band

    #print(f"Result saved to {output_path}")

# Example usage

def apply_fda_data(data, s_data): 
    """
    Applies FDA to data using style_data. 
    """
    f_data = data[0]
    num_bands = 4 #b,g,r,nir
    src = np.empty_like(data[0])
    target = np.empty_like(data[0])
    for i in range(num_bands):
        src_band = normalize_band(f_data[i])
        src[i] = src_band
        target_band = resize_band(src_band, s_data[i])
        target_band = normalize_band(target_band)
        target[i] = target_band

    # Apply FDA to bands
    fda_result = FDA_source_to_target_np(src, target, L=0.01)

    for i in range(num_bands):
        fda_result[i] = normalize_band(fda_result[i])

    # Ensure FDA result is in [0, 255] range
    fda_result = np.clip(fda_result, 0, 255).astype(np.uint8)

    return [fda_result, data[1]]

        
def resize_band(src, s_data):
    """
    Resizes s_data to the size of src.
    """
    reshaped = []
    if (s_data.shape != src.shape):
        for original_length, new_length in zip(s_data.shape, (src.shape)):
            reshaped.append(np.linspace(0, original_length-1, new_length))
        coords = np.meshgrid(*reshaped, indexing='ij')
        return map_coordinates(s_data, coords)
    

def normalize_band(band):
    band = band.astype(np.float64)
    if band.max() != band.min():
        band = ((band - band.min()) / (band.max() - band.min())) * 255
    else: 
        band = ((band - band.min())/ (1e-5)) * 255
    band = band.astype(np.uint8)
    return band


'''
source = "../../data/CN/tiles/features/tile_0000.tif"
target = "Style/000012.png"
output = "transfered_data/result.tif"

apply_fda_and_save(source, target, output)
'''