import os  # If directory manipulation is involved (e.g., path joining, existence checks)
import numpy as np  # For numerical operations like stacking arrays
import rasterio  # For reading .tif files
from scipy.ndimage import map_coordinates
from PIL import Image

class BigEarthDataProcessor:

    big_earth_dir = "../data/style/BigEarthNet-S2"
    ref_map_dir = "../data/style/ReferenceMaps"
    water_tif = "../data/style/Water"
    gen_style = "../data/style/all"

    big_earth_dirs = []
    ref_map_dirs = []

    def __init__(self): 
        self.get_big_earth_dirs()
        self.get_ref_map_dirs()
        self.create_style_tifs()

    def get_big_earth_dirs(self): 
        for root, dirs, files in os.walk(self.big_earth_dir):
            tif_dirs = {'path': '',
                        'filenames': files}
            prev_dirname = ''
            for name in files: 
                dirname = root.split(os.path.sep)[-1]
                if dirname != prev_dirname: 
                   tif_dirs['path'] = root
                   prev_dirname = tif_dirs['path']
            
            if tif_dirs['path'] != '' and tif_dirs['filenames'] != []: 
                self.big_earth_dirs.append(tif_dirs)

    def get_ref_map_dirs(self): 
        for root, dirs, files in os.walk(self.ref_map_dir):
            tif_dirs = {'path': '',
                        'filenames': files}
            prev_dirname = ''
            for name in files: 
                dirname = root.split(os.path.sep)[-1]
                if dirname != prev_dirname: 
                   tif_dirs['path'] = root
                   prev_dirname = tif_dirs['path']
            
            if tif_dirs['path'] != '' and tif_dirs['filenames'] != []: 
                self.ref_map_dirs.append(tif_dirs)
                        
    def create_style_tifs(self): 
        for i in range(len(self.big_earth_dirs)): 
            self.create_style_tif(self.big_earth_dirs[i], self.ref_map_dirs[i])

    def create_style_tif(self, big_earth_dirs, ref_map_dirs): 
        r, g, b, nir = self.get_rgb_nir_bands(big_earth_dirs)
        ref, transform = self.get_ref_band(ref_map_dirs)
        out_dir = self.get_out_dir(big_earth_dirs['path'], ref)
        self.write_style_tif(r, g, b, nir, out_dir, transform)

    def get_rgb_nir_bands(self, big_earth_dirs): 

        red_tif = self.get_band_filename_rgb(big_earth_dirs['filenames'], 'r')
        green_tif = self.get_band_filename_rgb(big_earth_dirs['filenames'], 'g')
        blue_tif = self.get_band_filename_rgb(big_earth_dirs['filenames'], 'b')
        
        nir_tifs = self.get_band_filename_nir(big_earth_dirs['filenames'])

        red_data = self.get_tif_data(big_earth_dirs['path'], red_tif)
        green_data = self.get_tif_data(big_earth_dirs['path'], green_tif)
        blue_data = self.get_tif_data(big_earth_dirs['path'], blue_tif)

        nir_data = self.build_nir_band(big_earth_dirs['path'], nir_tifs, red_data.shape)

        return red_data, green_data, blue_data, nir_data
                        

    def get_tif_data(self, path, filename, mask=False): 
        src = rasterio.open(path + os.path.sep + filename)
        data = src.read(1)
        if np.max(data) != np.min(data) and not mask:
            data = (data - np.min(data)) / ((np.max(data) - np.min(data)))*255
        return data
    
    def get_tif_transform(self, path, filename):
        src = rasterio.open(path + os.path.sep + filename)
        return src.transform

    def get_band_filename_rgb(self, filenames, option): 
        for file in filenames: 
            file_ending = file[-7:-4]
            if option=="r" and file_ending == 'B04': 
                return file
            elif option=="g" and file_ending == 'B03':
                return file
            elif option=='b' and file_ending == 'B02':
                return file
            
    def get_band_filename_nir(self, filenames): 
        nir_filenames = []
        for file in filenames: 
            file_ending = file[-7:-4]
            nir_endings = ['B05', 'B06', 'B07', 'B08', 'B8A']
            if file_ending in nir_endings: 
                nir_filenames.append(file)
        return nir_filenames

    def build_nir_band(self, path, filenames, shape):
        nir_data = np.empty(shape)
        for name in filenames: 
            new_data = self.get_tif_data(path, name)
            if (new_data.shape != shape):
                reshaped = []
                for original_length, new_length in zip(new_data.shape, (shape)):
                     reshaped.append(np.linspace(0, original_length-1, new_length))
                coords = np.meshgrid(*reshaped, indexing='ij')
                new_data = map_coordinates(new_data, coords)
            nir_data = np.add(nir_data, new_data)
        return nir_data


    def get_ref_band(self, ref_map_dirs): 

        ref_data = self.get_tif_data(ref_map_dirs['path'], ref_map_dirs['filenames'][0], mask=True)
        transform = self.get_tif_transform(ref_map_dirs['path'], ref_map_dirs['filenames'][0])
        return ref_data, transform
    

    def get_out_dir(self, path, ref): 
        path = path.split(os.path.sep) 
        name = path[-1]+".tif"
        path = path[0] + os.path.sep + path[1] + os.path.sep + path[2] + os.path.sep + "tiles"
        if self.contains_water(ref):
            out_dir = path + os.path.sep + "water" + os.path.sep + name
        else: 
            out_dir = path + os.path.sep + "general" + os.path.sep + name

        return out_dir


    def contains_water(self, ref): 
        water_categories = [212, 213, 331, 335, 411, 412, 421, 422, 423, 511, 512, 522, 523]

        values = np.unique(ref)

        for category in water_categories: 
            if category in values: 
                return True
        
        return False


    def write_style_tif(self, r, g, b, nir, out_dir, transform): 
        out_tif = np.stack([r, g, b, nir])
        __, height, width = out_tif.shape
        with rasterio.open(out_dir, 'w', 
                           width = width, 
                           height = height,
                           dtype=out_tif.dtype, 
                           count=4, 
                           transform=transform) as dst:
            dst.write(out_tif) 

        out_path = out_dir.split(os.path.sep)
        out_path = out_path[0:3] + ["images"] + out_path[4:]
        name = out_path[-1][:-4]+".png"
        out_path[-1] = name
        out_path = os.path.sep.join(out_path)
        get_sample_style_tile(out_dir, out_path)

        print(f"Style TIF created at: {out_dir}")

def main(): 
    BigEarthDataProcessor()

def get_sample_style_tile(path, out_path): 
    src = rasterio.open(path)
    r = src.read(1)
    g = src.read(2)
    b = src.read(3)
    img = np.uint8(np.stack([r, g, b], axis=2))
    im = Image.fromarray(img)
    im.save(out_path)

if __name__ == "__main__": 
    main()