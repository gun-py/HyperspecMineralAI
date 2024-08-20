import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
import umap

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

    def forward(self, x):
        return self.resnet(x)

class LandmarkFeatureExtractor:
    def __init__(self):
        self.feature_extractor = FeatureExtractor().eval()
        self.preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract_deep_features(self, image, landmarks):
        mouth_indices = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]
        mouth_landmarks = [landmarks[i] for i in mouth_indices]

        mouth_patch = image[min([p[1] for p in mouth_landmarks]):max([p[1] for p in mouth_landmarks]),
                            min([p[0] for p in mouth_landmarks]):max([p[0] for p in mouth_landmarks])]

        mouth_patch = self.preprocess(mouth_patch).unsqueeze(0)

        with torch.no_grad():
            features = self.feature_extractor(mouth_patch)
        
        return features.flatten().numpy()

class DimensionalityReducer:
    def __init__(self, method='tsne'):
        if method == 'tsne':
            self.reducer = TSNE(n_components=2, perplexity=30, learning_rate=200)
        elif method == 'umap':
            self.reducer = umap.UMAP(n_components=2)
        else:
            raise ValueError("Invalid method. Choose either 'tsne' or 'umap'.")

    def reduce_dimensions(self, features):
        return self.reducer.fit_transform(features)

class EmbeddingClusterer:
    def __init__(self, eps=0.5, min_samples=5):
        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples)

    def cluster_embeddings(self, embeddings):
        return self.dbscan.fit_predict(embeddings)

class FaceAnalysisPipeline:
    def __init__(self):
        self.landmark_extractor = LandmarkFeatureExtractor()
        self.reducer = DimensionalityReducer(method='umap')
        self.clusterer = EmbeddingClusterer()

    def process_image(self, image_path):
        image = cv2.imread(image_path)
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        detected_faces = DetectFaces(grayscale_image)
        all_features = []

        for face in detected_faces:
            landmarks = GetDeepFaceLandmarks(grayscale_image, face)
            features = self.landmark_extractor.extract_deep_features(grayscale_image, landmarks)
            all_features.append(features)

        all_features = np.array(all_features)
        embeddings = self.reducer.reduce_dimensions(all_features)
        clusters = self.clusterer.cluster_embeddings(embeddings)

        print("Embeddings:", embeddings)
        print("Cluster labels:", clusters)

pipeline = FaceAnalysisPipeline()
pipeline.process_image('input_image.jpg')
