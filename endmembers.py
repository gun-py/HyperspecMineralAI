import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter, median_filter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import cupy as cp
import logging
from numba import cuda
import cv2

logging.basicConfig(filename='ppi_calculation_advanced.log', level=logging.INFO, format='%(asctime)s - %(message)s')

class SpectralAnalyzer:
    def __init__(self, data):
        self.data = data

    def calculate_mahalanobis_distance(self, spectrum1, spectrum2, inv_cov_matrix):
        diff = spectrum1 - spectrum2
        distance = mahalanobis(diff, np.zeros_like(diff), inv_cov_matrix)
        return distance

    def calculate_pca(self, data, n_components=30):
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        pca = PCA(n_components=n_components)
        pca_data = pca.fit_transform(data_scaled)
        return pca_data, pca.components_, pca.mean_

class PPIProcessor:
    def __init__(self, hyperspectral_data, endmembers, use_gpu=False):
        self.hyperspectral_data = hyperspectral_data
        self.endmembers = endmembers
        self.use_gpu = use_gpu

    def initialize_zeros_matrix(self, dimensions):
        return np.zeros(dimensions, dtype=np.uint8)

    def initialize_spatial_filter_kernel(self, window_size):
        return np.ones((window_size, window_size), dtype=np.float32)

    def apply_gaussian_filter(self, input_matrix, sigma=1.0):
        return gaussian_filter(input_matrix, sigma=sigma)

    def apply_median_filter(self, input_matrix, size=3):
        return median_filter(input_matrix, size=size)

    def apply_spatial_filter(self, input_matrix, kernel, filter_type='median'):
        if filter_type == 'gaussian':
            return self.apply_gaussian_filter(input_matrix)
        elif filter_type == 'median':
            return self.apply_median_filter(input_matrix)
        else:
            raise ValueError("Unsupported filter type")

    def calculate_ppi(self, spectral_angle_threshold=0.1, window_size=5):
        logging.info("Starting PPI Calculation")

        preprocessor = SpectralAnalyzer(self.hyperspectral_data)
        data_reshaped = self.hyperspectral_data.reshape(-1, self.hyperspectral_data.shape[-1])
        pca_data, pca_components, pca_mean = preprocessor.calculate_pca(data_reshaped)
        pca_endmembers = preprocessor.calculate_pca(np.array(self.endmembers), n_components=pca_data.shape[1])[0]
        logging.info("PCA applied to hyperspectral data and endmembers")

        ppi_map = self.initialize_zeros_matrix(self.hyperspectral_data.shape[:2])
        inv_cov_matrix = np.linalg.inv(np.cov(pca_data.T))

        for i in range(self.hyperspectral_data.shape[0]):
            for j in range(self.hyperspectral_data.shape[1]):
                pixel_spectrum = self.hyperspectral_data[i, j, :]
                pixel_pca = np.dot(pixel_spectrum - pca_mean, pca_components.T)
                min_distance = np.inf
                for endmember in pca_endmembers:
                    endmember_pca = np.dot(endmember - pca_mean, pca_components.T)
                    distance = preprocessor.calculate_mahalanobis_distance(pixel_pca, endmember_pca, inv_cov_matrix)
                    if distance < min_distance:
                        min_distance = distance
                if min_distance <= spectral_angle_threshold:
                    ppi_map[i, j] = 1

        spatial_filter_kernel = self.initialize_spatial_filter_kernel(window_size)
        filtered_ppi_map = self.apply_spatial_filter(ppi_map, spatial_filter_kernel, filter_type='gaussian')

        self.save_image(filtered_ppi_map, "ppi_map_result_advanced.jpg")

        logging.info("PPI Calculation completed and result saved")
        return filtered_ppi_map

    def save_image(self, image_matrix, file_name):
        cv2.imwrite(file_name, image_matrix * 255)

if __name__ == '__main__':
    hyperspectral_data = load_hyperspectral_data('hyperion_data.e01')  # Placeholder for actual data loading function
    endmembers = load_endmembers('endmembers.txt')  # Placeholder for actual endmembers loading function
    ppi_processor = PPIProcessor(hyperspectral_data, endmembers, use_gpu=True)
    ppi_result = ppi_processor.calculate_ppi(spectral_angle_threshold=0.1, window_size=5)
    logging.info("PPI result processing completed")
