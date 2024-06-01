# Scalable ML Model with Real-Time Inference Endpoint for Customer Segmentation

![kuber_doc_seg](https://github.com/Abhi0323/Real-Time-Customer-Segmentation-with-Scalable-Kubernetes-Deployment-and-CI-CD-Integration/assets/112967999/099442ac-08ae-4dc7-92cd-79c871e25e35)

For a detailed blog explanation of the end-to-end project, visit this link: https://medium.com/@abhishekgoud1212/building-a-scalable-ml-model-with-real-time-inference-endpoint-for-customer-segmentation-e511cbec5bb3

## Overview:

This repository showcases the development of a scalable machine learning model for customer segmentation, capable of handling real-time inference through a REST API. The project tackles the business challenge of segmenting customers using unstructured data to create tailored marketing strategies. The solution incorporates complex workflows such as data ingestion, transformation, model training, and deployment, leveraging modern tools like FastAPI, Docker, Kubernetes, and GitHub Actions.

## Business Problem

Customer segmentation is vital for businesses to understand and address the diverse needs of their customer base. By grouping customers based on their behaviors and characteristics, companies can tailor marketing efforts, enhancing customer engagement and retention. This project develops a machine learning model that segments customers and provides real-time predictions via a REST API.

## Workflow

The project workflow includes several interconnected components:

* **1. Data Ingestion:** Load and prepare the customer dataset for analysis.
* **2. Data Transformation:** Clean and preprocess the data to make it suitable for model training.
* **3. Model Training:** Utilize the K-Means clustering algorithm to segment customers and determine the optimal number of clusters using the Elbow Method.
* **4. Real-Time Inference with FastAPI:** Develop a REST API using FastAPI to serve the model and provide real-time predictions.
* **5. Containerization with Docker:** Use Docker to containerize the FastAPI application, ensuring consistent deployment across various environments.
* **6. Deployment with Kubernetes:** Deploy the containerized application to a Kubernetes cluster for scalability and high availability.
* **7. CI/CD Pipeline with GitHub Actions:** Automate the build and deployment process using GitHub Actions, enabling continuous integration and continuous deployment.
* **8. Testing:** Verify the model's performance and accuracy using Postman for API endpoint testing and a Gradio interface hosted on Hugging Face Spaces for user-friendly testing.

## Project Structure

```
.
├── .github/workflows       # GitHub Actions workflows
│   └── deploy.yml
├── artifacts               # Directory for storing artifacts like models
├── src                     # Source code for the application
│   ├── components          # Data ingestion, transformation, and model training
│   ├── pipeline            # Prediction pipeline
│   ├── utils               # Utility functions
├── Dockerfile              # Docker configuration file
├── deployment.yaml         # Kubernetes deployment configuration
├── service.yaml            # Kubernetes service configuration
├── requirements.txt        # Python dependencies
├── main.py                 # FastAPI application
└── README.md               # Project description
```

## Key Technologies

* **FastAPI:** Framework for developing the REST API.
* **Docker:** Containerization tool for creating consistent deployment environments.
  <img width="1449" alt="Screenshot 2024-06-01 at 10 14 18 AM" src="https://github.com/Abhi0323/Real-Time-Customer-Segmentation-with-Scalable-Kubernetes-Deployment-and-CI-CD-Integration/assets/112967999/11968d08-ad8d-46b9-b6ae-365677918b5d">
* **Kubernetes:** Orchestrates deployment, scaling, and management of containerized applications.
  <img width="1460" alt="Screenshot 2024-06-01 at 10 15 47 AM" src="https://github.com/Abhi0323/Real-Time-Customer-Segmentation-with-Scalable-Kubernetes-Deployment-and-CI-CD-Integration/assets/112967999/3d91eda6-232d-4701-934c-91d2a90a10f8">
* **GitHub Actions:** Automates the CI/CD pipeline for continuous integration and deployment.
* **Azure:** Uses Azure Container Registry for storing Docker images and Azure Kubernetes Service for hosting the application.
  
## Testing

* **Postman:** Utilized for testing the API endpoints.
  
  <img width="1459" alt="Screenshot 2024-05-31 at 10 26 47 AM" src="https://github.com/Abhi0323/Real-Time-Customer-Segmentation-with-Scalable-Kubernetes-Deployment-and-CI-CD-Integration/assets/112967999/21156ce5-46d6-4291-bf65-99d6fad0a314">

* **Gradio Interface:** Developed a user-friendly interface hosted on Hugging Face Spaces to test the model in real-time.

  <img width="1449" alt="Screenshot 2024-05-31 at 5 40 12 PM" src="https://github.com/Abhi0323/Real-Time-Customer-Segmentation-with-Scalable-Kubernetes-Deployment-and-CI-CD-Integration/assets/112967999/a733ff45-df63-4866-96f1-7367f1c24708">

  
## Conclusion

This project provides an end-to-end solution for scalable customer segmentation with real-time inference capabilities. By leveraging modern tools and technologies, the model is robust, scalable, and easy to deploy, helping businesses tailor their marketing strategies effectively.
