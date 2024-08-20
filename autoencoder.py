import numpy as np
import cupy as cp
from sklearn.decomposition import DictionaryLearning
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import mahalanobis
from sklearn.manifold import TSNE, UMAP
from sklearn.decomposition import FastICA
from sklearn.decomposition import NMF
import dask.array as da
from dask.distributed import Client
import logging
from typing import List, Tuple, Dict
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

logging.basicConfig(filename='data_processing_advanced.log', level=logging.INFO, format='%(asctime)s - %(message)s')

class Autoencoder:
    def __init__(self, input_dim: int, encoding_dim: int):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.model = None
        self.encoder = None

    def build_model(self):
        input_layer = Input(shape=(self.input_dim,))
        encoded = Dense(self.encoding_dim, activation='relu')(input_layer)
        decoded = Dense(self.input_dim, activation='sigmoid')(encoded)
        autoencoder = Model(inputs=input_layer, outputs=decoded)
        self.encoder = Model(inputs=input_layer, outputs=encoded)
        autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
        self.model = autoencoder

    def train(self, data: np.ndarray, epochs: int = 50, batch_size: int = 256):
        self.model.fit(data, data, epochs=epochs, batch_size=batch_size, shuffle=True)

    def transform(self, data: np.ndarray) -> np.ndarray:
        return self.encoder.predict(data)

class HyperspectralDataProcessor:
    def __init__(self, file_path: str, use_gpu: bool = False):
        self.file_path = file_path
        self.dataset = None
        self.hyperspectral_data = None
        self.use_gpu = use_gpu
        if use_gpu:
            cp.cuda.Device(0).use()

    def open_dataset(self):
        logging.info(f"Opening dataset: {self.file_path}")
        self.dataset = np.load(self.file_path)
        self.hyperspectral_data = self.dataset['data']
        logging.info("Dataset opened successfully")

    def close_dataset(self):
        if self.dataset is not None:
            logging.info("Closing dataset")
            self.dataset = None

    def reshape_to_2D(self) -> np.ndarray:
        logging.info("Reshaping hyperspectral data to 2D")
        num_bands, height, width = self.hyperspectral_data.shape
        data2D = self.hyperspectral_data.reshape(num_bands, height * width).T
        return data2D

    def apply_autoencoder(self, data2D: np.ndarray, encoding_dim: int) -> np.ndarray:
        logging.info(f"Applying Autoencoder with encoding dimension {encoding_dim}")
        autoencoder = Autoencoder(input_dim=data2D.shape[1], encoding_dim=encoding_dim)
        autoencoder.build_model()
        autoencoder.train(data2D, epochs=50, batch_size=256)
        reduced_data = autoencoder.transform(data2D)
        return reduced_data

    def vertex_component_analysis(self, data2D: np.ndarray, num_endmembers: int) -> np.ndarray:
        logging.info(f"Performing Vertex Component Analysis with {num_endmembers} endmembers")
        endmembers = [data2D[0]]
        while len(endmembers) < num_endmembers:
            projections = self.calculate_projections(data2D, endmembers)
            max_residual_pixel = self.find_max_residual_pixel(data2D, projections, endmembers)
            endmembers.append(max_residual_pixel)
        return np.array(endmembers)

    def calculate_projections(self, data2D: np.ndarray, endmembers: List[np.ndarray]) -> np.ndarray:
        logging.info("Calculating projections")
        projections = [self.calculate_projection(pixel, endmembers) for pixel in data2D]
        return np.array(projections)

    def calculate_projection(self, pixel: np.ndarray, endmembers: List[np.ndarray]) -> np.ndarray:
        projection = np.array([np.dot(pixel, endmember) / np.dot(endmember, endmember) for endmember in endmembers])
        return projection

    def find_max_residual_pixel(self, data2D: np.ndarray, projections: np.ndarray, endmembers: List[np.ndarray]) -> np.ndarray:
        logging.info("Finding maximum residual pixel")
        max_residual_pixel = max(data2D, key=lambda pixel: self.calculate_residual(pixel, projections, endmembers))
        return max_residual_pixel

    def calculate_residual(self, pixel: np.ndarray, projection: np.ndarray, endmembers: List[np.ndarray]) -> float:
        reconstructed_pixel = np.sum(projection[:, np.newaxis] * np.array(endmembers), axis=0)
        residual = pixel - reconstructed_pixel
        return np.linalg.norm(residual)

    def apply_dictionary_learning(self, data2D: np.ndarray, n_components: int) -> np.ndarray:
        logging.info(f"Applying Dictionary Learning with {n_components} components")
        dict_learner = DictionaryLearning(n_components=n_components, alpha=1, max_iter=500)
        dict_learner.fit(data2D)
        return dict_learner.components_

    def apply_ica(self, data2D: np.ndarray, n_components: int) -> np.ndarray:
        logging.info(f"Applying ICA with {n_components} components")
        ica = FastICA(n_components=n_components)
        ica_data = ica.fit_transform(data2D)
        return ica_data

    def apply_nmf(self, data2D: np.ndarray, n_components: int) -> np.ndarray:
        logging.info(f"Applying NMF with {n_components} components")
        nmf = NMF(n_components=n_components)
        nmf_data = nmf.fit_transform(data2D)
        return nmf_data

    def apply_spatial_filter(self, data2D: np.ndarray, kernel_size: int) -> np.ndarray:
        logging.info(f"Applying spatial filter with kernel size {kernel_size}")
        filtered_data = np.array([self.median_filter(data, kernel_size) for data in data2D])
        return filtered_data

    def median_filter(self, data: np.ndarray, kernel_size: int) -> np.ndarray:
        from scipy.ndimage import median_filter
        return median_filter(data, size=kernel_size)

    def apply_tsne(self, data2D: np.ndarray, n_components: int) -> np.ndarray:
        logging.info(f"Applying t-SNE with {n_components} components")
        tsne = TSNE(n_components=n_components)
        tsne_data = tsne.fit_transform(data2D)
        return tsne_data

    def apply_umap(self, data2D: np.ndarray, n_components: int) -> np.ndarray:
        logging.info(f"Applying UMAP with {n_components} components")
        umap = UMAP(n_components=n_components)
        umap_data = umap.fit_transform(data2D)
        return umap_data

    def save_as_geoTIFF(self, data: np.ndarray, output_file_path: str) -> str:
        logging.info(f"Saving data as GeoTIFF: {output_file_path}")
        np.savez(output_file_path, data=data)
        return output_file_path

def main():
    input_file_path = 'your_hyperion_dataset.npz'
    output_file_path = 'filtered_and_reduced_data.npz'
    num_endmembers = 10

    processor = HyperspectralDataProcessor(input_file_path, use_gpu=True)
    processor.open_dataset()

    data2D = processor.reshape_to_2D()
    encoding_dim = 50
    reduced_data_autoencoder = processor.apply_autoencoder(data2D, encoding_dim)

    endmembers = processor.vertex_component_analysis(data2D, num_endmembers)

    n_components_dict = 15
    dictionary_components = processor.apply_dictionary_learning(data2D, n_components_dict)

    tsne_data = processor.apply_tsne(reduced_data_autoencoder, n_components=2)

    result_path = processor.save_as_geoTIFF(tsne_data, output_file_path)
    processor.close_dataset()

    print(f"Processed data saved to {result_path}")

if __name__ == '__main__':
    main()
