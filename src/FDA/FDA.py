import numpy as np
from PIL import Image
from .utils import FDA_source_to_target_np
import rasterio

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

# Example usage
'''
source = "../../data/CN/tiles/features/tile_0000.tif"
target = "Style/000012.png"
output = "transfered_data/result.tif"

apply_fda_and_save(source, target, output)
'''