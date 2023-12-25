# Image Recommendation System using TensorFlow

This project utilizes TensorFlow 2.10 and Scikit-learn 1.3.0 to create an image recommendation system. The system is designed to take an input image from the user and recommend five similar images from the Fashion Product Images Dataset available on [Kaggle](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset).

## Purpose

The primary goal of this project is to explore Convolutional Neural Networks (CNNs) and deep learning techniques. It specifically utilizes the ResNet-50 model, a pre-trained model from TensorFlow that was originally trained on the ImageNet dataset. By leveraging this pre-trained model, the project aims to understand how to use transfer learning for image recognition tasks.

## Features

- Utilizes the ResNet-50 pre-trained model for image recognition.
- Recommends five images similar to the input image from the Fashion Product Images Dataset.
- Utilizes Scikit-learn for calculating nearest neighbors for recommendation.

## Target Audience

This project is beneficial for developers and machine learning enthusiasts looking to understand and implement image recommendation systems using pre-trained CNN models.

## Technical Details

- Python 3.9.0 is used for development.
- TensorFlow 2.10, Scikit-learn 1.3.0, and the ResNet-50 model from TensorFlow are the primary dependencies.
- Two pickle files trained on the Fashion Product Images Dataset are required for the system to recommend similar images.

## Installation and Usage

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your_username/your_project.git
   cd your_project
