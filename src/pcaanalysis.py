import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class PCAAnalysis:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def load_images(self):
        """Load and preprocess images."""
        images = []
        for file_name in os.listdir(self.dataset_path):
            img = Image.open(os.path.join(self.dataset_path, file_name))
            img = img.resize((64, 64), Image.BILINEAR)
            images.append(np.array(img))
        self.images = np.array(images, dtype=np.float32)
        self.images_flattened = self.images.reshape(self.images.shape[0], -1, 3)
        print(f"Loaded images with shape: {self.images.shape}")

    def perform_pca(self, X):
        """Perform PCA on the given data."""
        mean = np.mean(X, axis=0)
        X_centered = X - mean
        covariance_matrix = np.dot(X_centered.T, X_centered) / (X.shape[0] - 1)
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        sorted_indices = np.argsort(-eigenvalues)
        return eigenvalues[sorted_indices], eigenvectors[:, sorted_indices], mean

    def run(self):
        """Main method to execute PCA analysis."""
        self.load_images()
        channels = ["Red", "Green", "Blue"]
        pve_results = {}

        for i, color in enumerate(channels):
            X_channel = self.images_flattened[:, :, i]
            eigenvalues, eigenvectors, mean = self.perform_pca(X_channel)
            pve = eigenvalues / np.sum(eigenvalues)
            cumulative_pve = np.cumsum(pve)

            print(f"{color} Channel:")
            print(f"Proportion of Variance Explained (PVE): {pve[:10]}")
            print(f"Cumulative PVE: {cumulative_pve[:10]}")
            pve_results[color] = cumulative_pve

            # Reconstruct and visualize the first image
            k_values = [1, 50, 250, 500, 1000, X_channel.shape[1]]
            for k in k_values:
                top_k_eigenvectors = eigenvectors[:, :k]
                projection = np.dot((X_channel[0] - mean), top_k_eigenvectors)
                reconstruction = np.dot(projection, top_k_eigenvectors.T) + mean
                plt.imshow(reconstruction.reshape(64, 64), cmap="gray")
                plt.title(f"Reconstruction with k={k} components")
                plt.show()
