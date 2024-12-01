import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import seaborn as sns

class PCAAnalysis:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def load_images(self):
        """Load and preprocess images."""
        self.image_data = []
        for file_name in sorted(os.listdir(self.dataset_path)):
            if file_name.endswith('.png'):  # Process only PNG files
                img = Image.open(os.path.join(self.dataset_path, file_name))
                img = img.resize((64, 64), Image.BILINEAR)  # Resize to 64x64
                img_array = np.array(img, dtype=np.float32)
                self.image_data.append(img_array)
        
        self.image_data = np.array(self.image_data)
        self.image_data_flattened = self.image_data.reshape(len(self.image_data), -1, 3)
        print(f"Loaded images: {len(self.image_data)}")
    
    def split_rgb_channels(self):
        """Split RGB channels into separate data structures."""
        print("Splitting RGB channels...")
        self.red_channel = self.image_data_flattened[:, :, 0]
        self.green_channel = self.image_data_flattened[:, :, 1]
        self.blue_channel = self.image_data_flattened[:, :, 2]
        print("RGB split complete.")

    def compute_pca(self, data):
        """Perform PCA and return sorted eigenvalues and eigenvectors."""
        mean = np.mean(data, axis=0)
        centered_data = data - mean
        covariance_matrix = np.cov(centered_data, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        return eigenvalues[sorted_indices], eigenvectors[:, sorted_indices], mean

    def calculate_pve(self, eigenvalues):
        """Calculate PVE and cumulative PVE."""
        total_variance = np.sum(eigenvalues)
        pve = eigenvalues / total_variance
        cumulative_pve = np.cumsum(pve)
        return pve, cumulative_pve

    def display_pca_results(self, eigenvalues, pve, cumulative_pve, color_name):
        """Display the top 10 eigenvalues and PVE results for a color channel."""
        table_data = {
            "PC Number": np.arange(1, 12),
            "Eigenvalue": np.round(eigenvalues[:11], 2),
            "PVE": np.round(pve[:11], 3),
            "Cumulative PVE": np.round(cumulative_pve[:11], 3),
        }
        table_df = pd.DataFrame(table_data)
        print(f"\n{color_name} Values (X_{color_name[0]}):\n")
        print(table_df)
        print(f"Total PVE of top 10 principal components for {color_name}: {np.sum(pve[:10]):.3f}")
        print(
            f"Total PVE of top 11 principal components for {color_name}: {cumulative_pve[10]:.3f}\n"
        )

    def visualize_top_principal_components(self):
        """Visualize the top 10 principal components for RGB channels."""
        print("Visualizing top 10 principal components for each channel...")
        reshaped_red = self.red_eigenvectors[:, :10].T.reshape(10, 64, 64)
        reshaped_green = self.green_eigenvectors[:, :10].T.reshape(10, 64, 64)
        reshaped_blue = self.blue_eigenvectors[:, :10].T.reshape(10, 64, 64)

        normalized_R = np.array([self.normalize(pc) for pc in reshaped_red])
        normalized_G = np.array([self.normalize(pc) for pc in reshaped_green])
        normalized_B = np.array([self.normalize(pc) for pc in reshaped_blue])

        rgb_images = np.stack((normalized_R, normalized_G, normalized_B), axis=-1)  # Shape: (10, 64, 64, 3)

        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        for i, ax in enumerate(axes.flat):
            ax.imshow(rgb_images[i])
            ax.set_title(f"PC {i + 1}")
            ax.axis("off")
        plt.tight_layout()
        plt.show()

    def normalize(self, matrix):
        """Normalize a matrix for visualization."""
        min_val = np.min(matrix)
        max_val = np.max(matrix)
        return (matrix - min_val) / (max_val - min_val)

    def reconstruct_image(self, channel_data, eigenvalues, eigenvectors, mean, k):
        """Reconstruct the image using the top k principal components."""
        top_k_vectors = eigenvectors[:, :k]
        projection = np.dot(channel_data - mean, top_k_vectors)
        reconstruction = np.dot(projection, top_k_vectors.T) + mean
        return reconstruction

    def visualize_reconstruction(self, k_values):
        """Visualize reconstructed images."""
        for k in k_values:
            red_reconstructed = self.reconstruct_image(
                self.red_channel[0], self.red_eigenvalues, self.red_eigenvectors, self.red_mean, k
            ).reshape(64, 64)
            green_reconstructed = self.reconstruct_image(
                self.green_channel[0], self.green_eigenvalues, self.green_eigenvectors, self.green_mean, k
            ).reshape(64, 64)
            blue_reconstructed = self.reconstruct_image(
                self.blue_channel[0], self.blue_eigenvalues, self.blue_eigenvectors, self.blue_mean, k
            ).reshape(64, 64)

            reconstructed_image = np.stack([red_reconstructed, green_reconstructed, blue_reconstructed], axis=-1)

            plt.imshow(reconstructed_image / 255)
            plt.title(f"Reconstruction with {k} components")
            plt.axis("off")
            plt.show()

    def run(self):
        """Run the PCA analysis and visualize results."""
        self.load_images()
        self.split_rgb_channels()

        # Perform PCA for the Red channel
        print("Performing PCA on Red Channel...")
        self.red_eigenvalues, self.red_eigenvectors, self.red_mean = self.compute_pca(self.red_channel)
        red_pve, red_cumulative_pve = self.calculate_pve(self.red_eigenvalues)
        self.display_pca_results(self.red_eigenvalues, red_pve, red_cumulative_pve, "Red")

        # Perform PCA for the Green channel
        print("Performing PCA on Green Channel...")
        self.green_eigenvalues, self.green_eigenvectors, self.green_mean = self.compute_pca(self.green_channel)
        green_pve, green_cumulative_pve = self.calculate_pve(self.green_eigenvalues)
        self.display_pca_results(self.green_eigenvalues, green_pve, green_cumulative_pve, "Green")

        # Perform PCA for the Blue channel
        print("Performing PCA on Blue Channel...")
        self.blue_eigenvalues, self.blue_eigenvectors, self.blue_mean = self.compute_pca(self.blue_channel)
        blue_pve, blue_cumulative_pve = self.calculate_pve(self.blue_eigenvalues)
        self.display_pca_results(self.blue_eigenvalues, blue_pve, blue_cumulative_pve, "Blue")

        # Visualize top 10 principal components
        self.visualize_top_principal_components()

        # Reconstruction and visualization
        k_values = [1, 50, 250, 500, 1000, 4096]
        self.visualize_reconstruction(k_values)

dataset_path = "fake"

# Perform PCA Analysis
pca = PCAAnalysis(dataset_path)
pca.run()

