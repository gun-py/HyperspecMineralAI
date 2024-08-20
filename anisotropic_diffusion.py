import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from skimage.restoration import denoise_bilateral
from skimage.filters import anisotropic_diffusion
import h5py
import logging
from typing import Tuple, Dict
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS
import cupy as cp

class HyperspectralDataProcessor:
    def __init__(self, file_path: str, use_gpu=False):
        self.file_path = file_path
        self.dataset = None
        self.hyperspectral_data = None
        self.use_gpu = use_gpu
        if use_gpu:
            cp.cuda.Device(0).use()  # Initialize GPU

    def open_dataset(self):
        logging.info(f"Opening dataset: {self.file_path}")
        self.dataset = h5py.File(self.file_path, 'r')  # Assuming HDF5 format for data
        self.hyperspectral_data = self.dataset['data'][:]  # Adjust based on actual data structure
        logging.info("Dataset opened successfully")

    def close_dataset(self):
        if self.dataset:
            self.dataset.close()
            logging.info("Dataset closed")

    def reshape_to_2D(self) -> np.ndarray:
        logging.info("Reshaping hyperspectral data to 2D")
        num_bands, height, width = self.hyperspectral_data.shape
        data2D = self.hyperspectral_data.reshape(num_bands, height * width).T
        return data2D

    def apply_pca(self, data2D: np.ndarray, n_components: int) -> np.ndarray:
        logging.info(f"Applying PCA with {n_components} components")
        scaler = StandardScaler()
        data2D_scaled = scaler.fit_transform(data2D)
        pca = PCA(n_components=n_components)
        reduced_data = pca.fit_transform(data2D_scaled)
        return reduced_data

    def apply_tsne(self, reduced_data: np.ndarray, n_components: int) -> np.ndarray:
        logging.info(f"Applying t-SNE with {n_components} components")
        tsne = TSNE(n_components=n_components, perplexity=30, n_iter=300)
        tsne_data = tsne.fit_transform(reduced_data)
        return tsne_data

    def apply_bilateral_filtering(self, data: np.ndarray, sigma_color: float, sigma_spatial: float) -> np.ndarray:
        logging.info(f"Applying bilateral filtering with sigma_color={sigma_color} and sigma_spatial={sigma_spatial}")
        num_bands, height, width = data.shape
        filtered_data = np.zeros_like(data)
        for band_index in range(num_bands):
            band_data = data[band_index]
            filtered_data[band_index] = denoise_bilateral(band_data, sigma_color=sigma_color, sigma_spatial=sigma_spatial)
        return filtered_data

    def apply_anisotropic_diffusion(self, data: np.ndarray, num_iter: int = 10) -> np.ndarray:
        logging.info(f"Applying anisotropic diffusion with num_iter={num_iter}")
        num_bands, height, width = data.shape
        diffused_data = np.zeros_like(data)
        for band_index in range(num_bands):
            band_data = data[band_index]
            diffused_data[band_index] = anisotropic_diffusion(band_data, niter=num_iter)
        return diffused_data

    def save_as_geotiff(self, filtered_data: np.ndarray, output_file_path: str, metadata: Dict) -> str:
        logging.info(f"Saving filtered data as GeoTIFF: {output_file_path}")
        num_bands, height, width = filtered_data.shape

        transform = from_origin(metadata['x_min'], metadata['y_max'], metadata['pixel_size_x'], metadata['pixel_size_y'])
        crs = CRS.from_epsg(metadata['epsg_code'])

        with rasterio.open(output_file_path, 'w', driver='GTiff', height=height, width=width,
                           count=num_bands, dtype='float32', crs=crs, transform=transform) as dst:
            for band_index in range(num_bands):
                dst.write(filtered_data[band_index], band_index + 1)
        logging.info("GeoTIFF file saved successfully")
        return output_file_path

def main():
    input_file_path = 'dataset.h5'
    output_file_path = 'filtered_and_reduced_data.tif'
    metadata = {
        'x_min': 0, 
        'y_max': 0, 
        'pixel_size_x': 1, 
        'pixel_size_y': 1, 
        'epsg_code': 4326
    }
    
    processor = HyperspectralDataProcessor(input_file_path, use_gpu=True)
    processor.open_dataset()
    
    data2D = processor.reshape_to_2D()
    n_components = 10
    reduced_data = processor.apply_pca(data2D, n_components)
    tsne_data = processor.apply_tsne(reduced_data, n_components=3)
    
    sigma_color = 0.1
    sigma_spatial = 15
    filtered_data_bilateral = processor.apply_bilateral_filtering(reduced_data.reshape(n_components, -1).reshape(n_components, 100, 100), sigma_color, sigma_spatial)
    
    filtered_data_aniso = processor.apply_anisotropic_diffusion(filtered_data_bilateral)
    
    result_path = processor.save_as_geotiff(filtered_data_aniso, output_file_path, metadata)
    processor.close_dataset()
    print(f"Filtered and reduced data saved to {result_path}")

if __name__ == '__main__':
    main()
