import os
import sys

from dataclasses import dataclass
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.components.data_transformation import DataTransformetaion

@dataclass
class ModelConfig:
    trained_model_path: str = os.path.join("artifacts", "model.pkl")
    elbow_plot_path: str = os.path.join("artifacts", "elbow_plot.png")

class ModelTrainer:
    def __init__(self):
        self.model_config = ModelConfig()

    def find_optimal_clusters(self, data, max_k=10):
        wcss = []  # Within-cluster sum of squares

        for i in range(1, max_k + 1):
            kmeans = KMeans(n_clusters=i, random_state=42)
            kmeans.fit(data)
            wcss.append(kmeans.inertia_)

        plt.figure(figsize=(10, 8))
        plt.plot(range(1, max_k + 1), wcss, marker='o', linestyle='--')
        plt.title('Elbow Method For Optimal k')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.savefig(self.model_config.elbow_plot_path)
        plt.close()

        # Finding the elbow point (optimal number of clusters)
        deltas = np.diff(wcss)
        second_deltas = np.diff(deltas)
        optimal_k = np.argmax(second_deltas) + 2  # Adding 2 because np.diff reduces the array size by 1 each time

        return optimal_k

    def initiate_model_training(self, data_array):
        try:
            logging.info("Finding the optimal number of clusters")
            optimal_clusters = self.find_optimal_clusters(data_array)
            logging.info(f"Optimal number of clusters found: {optimal_clusters}")

            logging.info("Training K-Means model with optimal number of clusters")
            kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
            kmeans.fit(data_array)

            logging.info(f"Saving the trained model to {self.model_config.trained_model_path}")
            save_object(file_path=self.model_config.trained_model_path, obj=kmeans)

            return kmeans.labels_
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == '__main__':
    data_path = '/Users/abhi/Desktop/ML-Docker-Kubernetes/notebook/data/Customer Data.csv'
    data_transformation = DataTransformetaion()
    transformed_data, processor_path = data_transformation.initiate_transformation(data_path)

    model_trainer = ModelTrainer()
    labels = model_trainer.initiate_model_training(transformed_data)
    print("Clustering labels:", labels)
