import numpy as np
import rasterio
from math import radians, degrees
from Py6S import *
import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.cluster import DBSCAN
import umap

class HyperionDataLoader:
    def __init__(self, data_path):
        self.data_path = data_path

    def load_data(self):
        with rasterio.open(self.data_path) as src:
            hyperion_data = src.read()
        return hyperion_data

    def get_wavelengths(self, hyperion_data):
        return np.linspace(400, 2500, hyperion_data.shape[0])

class AtmosphericCorrection:
    def __init__(self, atmospheric_parameters):
        self.atmospheric_parameters = atmospheric_parameters

    def calculate_raa(self, sza, vza):
        sza_deg = degrees(sza)
        vza_deg = degrees(vza)

        if sza_deg < vza_deg:
            raa = vza_deg - sza_deg
        else:
            raa = sza_deg - vza_deg

        if raa > 180:
            raa = 360 - raa

        return raa

    def apply_correction(self, hyperion_data, wavelengths, sza, vza, raa):
        s = SixS()
        s.geometry = Geometry.User()
        s.geometry.solar_z = degrees(sza)
        s.geometry.view_z = degrees(vza)
        s.geometry.solar_a = 0
        s.geometry.view_a = raa

        s.atmos_profile = AtmosProfile.PredefinedType(AtmosProfile.Tropical)
        s.aot550 = self.atmospheric_parameters['AOT']
        s.ground_reflectance = GroundReflectance.HomogeneousLambertian(0.3)

        surface_reflectance = np.zeros_like(hyperion_data, dtype=np.float32)

        for i, wavelength in enumerate(wavelengths):
            s.wavelength = Wavelength(wavelength)
            s.run()

            toa_radiance = hyperion_data[i]
            surface_reflectance[i] = (toa_radiance - s.outputs.atmospheric_intrinsic_radiance) / s.outputs.transmittance_total_scattering

        return surface_reflectance

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

    def forward(self, x):
        return self.resnet(x)

class ImageFeatureExtractor:
    def __init__(self):
        self.feature_extractor = FeatureExtractor().eval()
        self.preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract_features(self, image):
        image = self.preprocess(image).unsqueeze(0)

        with torch.no_grad():
            features = self.feature_extractor(image)

        return features.flatten().numpy()

class DimensionalityReducer:
    def __init__(self, method='umap'):
        if method == 'umap':
            self.reducer = umap.UMAP(n_components=2, random_state=42)
        else:
            raise ValueError("Unsupported method")

    def reduce_dimensions(self, features):
        return self.reducer.fit_transform(features)

class EmbeddingClusterer:
    def __init__(self, eps=0.5, min_samples=5):
        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples)

    def cluster_embeddings(self, embeddings):
        return self.dbscan.fit_predict(embeddings)

class HyperionAnalysisPipeline:
    def __init__(self, hyperion_data_path, atmospheric_parameters):
        self.data_loader = HyperionDataLoader(hyperion_data_path)
        self.atmospheric_correction = AtmosphericCorrection(atmospheric_parameters)
        self.feature_extractor = ImageFeatureExtractor()
        self.reducer = DimensionalityReducer(method='umap')
        self.clusterer = EmbeddingClusterer()

    def process(self):
        hyperion_data = self.data_loader.load_data()
        wavelengths = self.data_loader.get_wavelengths(hyperion_data)

        sza = radians(30)
        vza = radians(10)

        raa = self.atmospheric_correction.calculate_raa(sza, vza)

        surface_reflectance = self.atmospheric_correction.apply_correction(
            hyperion_data, wavelengths, sza, vza, raa
        )

        example_image = np.random.rand(224, 224, 3)
        features = self.feature_extractor.extract_features(example_image)

        embeddings = self.reducer.reduce_dimensions(features.reshape(1, -1))

        clusters = self.clusterer.cluster_embeddings(embeddings)

        print("Embeddings:", embeddings)
        print("Cluster labels:", clusters)

atmospheric_parameters = {
    "AOT": 0.2,
    "PWV": 2.0,
    "O3": 0.3,
}

pipeline = HyperionAnalysisPipeline("path_to_hyperion_data.hdr", atmospheric_parameters)
pipeline.process()
