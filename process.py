import numpy as np
import cupy as cp
from sklearn.decomposition import PCA, DictionaryLearning
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.layers import Input, Dense, Dropout, Conv1D, MaxPooling1D, Concatenate, Multiply, Activation
from tensorflow.keras.models import Model
from skimage.restoration import denoise_bilateral
from skimage.filters import anisotropic_diffusion
import logging
import h5py


class Autoencoder:
    def __init__(self, input_dim, encoding_dim):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.model = None
        self.encoder = None

    def build_model(self):
        input_layer = Input(shape=(self.input_dim,))
        encoded = Dense(self.encoding_dim, activation='relu')(input_layer)
        decoded = Dense(self.input_dim, activation='sigmoid')(encoded)
        self.model = Model(inputs=input_layer, outputs=decoded)
        self.encoder = Model(inputs=input_layer, outputs=encoded)
        self.model.compile(optimizer='adam', loss='binary_crossentropy')

    def train(self, data, epochs=50, batch_size=256):
        self.model.fit(data, data, epochs=epochs, batch_size=batch_size, shuffle=True)

    def transform(self, data):
        return self.encoder.predict(data)


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
        self.dataset = h5py.File(self.file_path, 'r') 
        self.hyperspectral_data = self.dataset['data'][:]
        logging.info("Dataset opened successfully")

    def close_dataset(self):
        if self.dataset:
            self.dataset.close()
            logging.info("Dataset closed")

    def reshape_to_2D(self) -> np.ndarray:
        logging.info("Reshaping hyperspectral data to 2D")
        num_bands = self.hyperspectral_data.shape[0]
        height = self.hyperspectral_data.shape[1]
        width = self.hyperspectral_data.shape[2]
        data2D = self.hyperspectral_data.reshape(num_bands, height * width).T
        return data2D

    def apply_pca(self, data2D: np.ndarray, n_components: int) -> np.ndarray:
        logging.info(f"Applying PCA with {n_components} components")
        scaler = StandardScaler()
        data2D_scaled = scaler.fit_transform(data2D)
        pca = PCA(n_components=n_components)
        reduced_data = pca.fit_transform(data2D_scaled)
        return reduced_data

    def apply_autoencoder(self, data2D: np.ndarray, encoding_dim: int) -> np.ndarray:
        logging.info(f"Applying Autoencoder with encoding dimension {encoding_dim}")
        autoencoder = Autoencoder(input_dim=data2D.shape[1], encoding_dim=encoding_dim)
        autoencoder.build_model()
        autoencoder.train(data2D, epochs=50, batch_size=256)
        reduced_data = autoencoder.transform(data2D)
        return reduced_data

    def apply_bilateral_filtering(self, data: np.ndarray, sigma_color: float, sigma_spatial: float) -> np.ndarray:
        logging.info(f"Applying bilateral filtering with sigma_color={sigma_color} and sigma_spatial={sigma_spatial}")
        num_bands = data.shape[0]
        height = data.shape[1]
        width = data.shape[2]
        filtered_data = np.zeros_like(data)
        for band_index in range(num_bands):
            band_data = data[band_index]
            filtered_data[band_index] = denoise_bilateral(band_data, sigma_color=sigma_color, sigma_spatial=sigma_spatial)
        return filtered_data

    def apply_anisotropic_diffusion(self, data: np.ndarray, num_iter: int = 10) -> np.ndarray:
        logging.info(f"Applying anisotropic diffusion with num_iter={num_iter}")
        num_bands = data.shape[0]
        height = data.shape[1]
        width = data.shape[2]
        diffused_data = np.zeros_like(data)
        for band_index in range(num_bands):
            band_data = data[band_index]
            diffused_data[band_index] = anisotropic_diffusion(band_data, niter=num_iter)
        return diffused_data

    def save_as_geotiff(self, filtered_data: np.ndarray, output_file_path: str, metadata: dict) -> str:
        logging.info(f"Saving filtered data as GeoTIFF: {output_file_path}")
        num_bands = filtered_data.shape[0]
        height = filtered_data.shape[1]
        width = filtered_data.shape[2]

        transform = from_origin(metadata['x_min'], metadata['y_max'], metadata['pixel_size_x'], metadata['pixel_size_y'])
        crs = CRS.from_epsg(metadata['epsg_code'])

        with rasterio.open(output_file_path, 'w', driver='GTiff', height=height, width=width,
                           count=num_bands, dtype='float32', crs=crs, transform=transform) as dst:
            for band_index in range(num_bands):
                dst.write(filtered_data[band_index], band_index + 1)
        logging.info("GeoTIFF file saved successfully")
        return output_file_path

def create_pairs(data: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pairs = []
    pair_labels = []
    num_samples = data.shape[0]
    
    for i in range(num_samples):
        for j in range(num_samples):
            pairs.append([data[i], data[j]])
            if labels[i] == labels[j]:
                pair_labels.append(1)  # Similar pair
            else:
                pair_labels.append(0)  # Dissimilar pair
    
    return np.array(pairs), np.array(pair_labels)