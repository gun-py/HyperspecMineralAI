import numpy as np
from sklearn.decomposition import SparsePCA, KernelPCA
from sklearn.covariance import LedoitWolf
from numba import jit, cuda
import cupy as cp
import logging
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class DataPreprocessor:
    def __init__(self, data):
        self.data = data

    def reshape_data(self):
        return self.data.reshape(-1, self.data.shape[-1])

    def calculate_mean_spectrum(self):
        return np.mean(self.data, axis=0)

class CovarianceCalculator:
    def __init__(self, data):
        self.data = data

    def calculate_covariance_matrix(self, method='ledoit'):
        if method == 'ledoit':
            cov_matrix = LedoitWolf().fit(self.data).covariance_
        else:
            cov_matrix = np.cov(self.data, rowvar=False)
        return cov_matrix

class DimensionalityReducer:
    def __init__(self, data):
        self.data = data

    def apply_sparse_pca(self, n_components=30):
        spca = SparsePCA(n_components=n_components, alpha=1, random_state=42)
        spca_data = spca.fit_transform(self.data)
        return spca_data, spca.components_

    def apply_kernel_pca(self, n_components=30, kernel='rbf'):
        kpca = KernelPCA(n_components=n_components, kernel=kernel, gamma=0.1, fit_inverse_transform=True)
        kpca_data = kpca.fit_transform(self.data)
        return kpca_data, kpca.alphas_, kpca.lambdas_

class FeatureExtractor:
    def __init__(self, data):
        self.data = data

    def deep_feature_extraction(self, epochs=10):
        model = Sequential([
            Dense(128, activation='relu', input_shape=(self.data.shape[1],)),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(16, activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        model.fit(self.data, self.data, epochs=epochs, batch_size=64, verbose=1)
        return model

    def apply_deep_feature_extraction(self, model):
        return model.predict(self.data)

class MNFTransform:
    def __init__(self, data):
        self.data = data

    def calculate_eigenvalues_and_eigenvectors(self, cov_matrix):
        eigenvalues, eigenvectors = eigh(cov_matrix)
        return eigenvalues, eigenvectors

    def sort_eigenvalues_and_eigenvectors(self, eigenvalues, eigenvectors):
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]
        return sorted_eigenvalues, sorted_eigenvectors

    def choose_mnf_components(self, eigenvalues, threshold=0.01):
        cumulative_variance = np.cumsum(eigenvalues) / np.sum(eigenvalues)
        n_components = np.argmax(cumulative_variance >= (1 - threshold)) + 1
        return n_components

    def extract_mnf_components(self, eigenvectors, n_components):
        return eigenvectors[:, :n_components]

    @staticmethod
    @jit(nopython=True)
    def compute_mnf_transform(data, mnf_eigenvectors, mean_spectrum):
        data_centered = data - mean_spectrum
        mnf_data = np.dot(data_centered, mnf_eigenvectors)
        return mnf_data

    @staticmethod
    @cuda.jit
    def compute_mnf_transform_gpu(data, mnf_eigenvectors, mnf_data):
        i = cuda.grid(1)
        if i < data.shape[0]:
            mnf_data[i] = cp.dot(data[i] - cp.mean(data, axis=0), mnf_eigenvectors)

    @staticmethod
    def inverse_mnf_transform(mnf_data, mnf_eigenvectors, mean_spectrum):
        reconstructed_data = np.dot(mnf_data, mnf_eigenvectors.T) + mean_spectrum
        return reconstructed_data

    @staticmethod
    def reshape_to_original_shape(data, original_shape):
        return data.reshape(original_shape)

class MNFProcessor:
    def __init__(self, hyperspectral_data, use_gpu=False):
        self.hyperspectral_data = hyperspectral_data
        self.use_gpu = use_gpu

    def process(self):
        logging.info("Starting MNF Transform")

        preprocessor = DataPreprocessor(self.hyperspectral_data)
        data_reshaped = preprocessor.reshape_data()
        mean_spectrum = preprocessor.calculate_mean_spectrum()
        logging.info("Data reshaped and mean spectrum calculated")

        feature_extractor = FeatureExtractor(data_reshaped)
        deep_model = feature_extractor.deep_feature_extraction()
        data_deep_features = feature_extractor.apply_deep_feature_extraction(deep_model)
        logging.info("Deep feature extraction applied")

        reducer = DimensionalityReducer(data_deep_features)
        pca_data, pca_components = reducer.apply_sparse_pca()
        kpca_data, kpca_alphas, kpca_lambdas = reducer.apply_kernel_pca()
        logging.info("Dimensionality reduction applied using PCA and Kernel PCA")

        cov_calculator = CovarianceCalculator(kpca_data)
        cov_matrix = cov_calculator.calculate_covariance_matrix()
        cov_matrix_regularized = cov_calculator.calculate_covariance_matrix(method='ledoit')
        logging.info("Covariance matrix calculated and regularized")

        mnf = MNFTransform(kpca_data)
        eigenvalues, eigenvectors = mnf.calculate_eigenvalues_and_eigenvectors(cov_matrix_regularized)
        sorted_eigenvalues, sorted_eigenvectors = mnf.sort_eigenvalues_and_eigenvectors(eigenvalues, eigenvectors)
        n_mnf_components = mnf.choose_mnf_components(sorted_eigenvalues)
        mnf_eigenvectors = mnf.extract_mnf_components(sorted_eigenvectors, n_mnf_components)
        logging.info(f"{n_mnf_components} MNF components selected and extracted")

        if self.use_gpu and data_reshaped.shape[0] > 1e6:
            data_gpu = cp.array(data_reshaped)
            mnf_eigenvectors_gpu = cp.array(mnf_eigenvectors)
            mnf_data_gpu = cp.empty((data_reshaped.shape[0], n_mnf_components), dtype=cp.float32)
            MNFTransform.compute_mnf_transform_gpu[(32,), (32,)](data_gpu, mnf_eigenvectors_gpu, mnf_data_gpu)
            mnf_data = cp.asnumpy(mnf_data_gpu)
            logging.info("MNF Transform computed using GPU")
        else:
            mnf_data = MNFTransform.compute_mnf_transform(data_reshaped, mnf_eigenvectors, mean_spectrum)
            logging.info("MNF Transform computed using CPU")

        denoised_data = MNFTransform.inverse_mnf_transform(mnf_data, mnf_eigenvectors, mean_spectrum)
        denoised_data = MNFTransform.reshape_to_original_shape(denoised_data, self.hyperspectral_data.shape)
        logging.info("Inverse MNF Transform applied and data reshaped to original dimensions")

        return denoised_data

if __name__ == '__main__':
    hyperspectral_data = load_hyperspectral_data('hyperion_data.e01')  
    processor = MNFProcessor(hyperspectral_data, use_gpu=True)
    result = processor.process()
    save_denoised_data(result, 'denoised_hyperion_data.e01') 
