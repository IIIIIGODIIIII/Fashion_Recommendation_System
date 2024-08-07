# Image Recommendation System using TensorFlow

This project utilizes TensorFlow 2.10, Scikit-learn, numpy, pickle, tqdm to create an image recommendation system. The system is designed to take an input image from the user and recommend five similar images from the Fashion Product Images Dataset available on [Kaggle](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset).

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
- TensorFlow 2.10, Scikit-learn 1.3.0, Numpy, Pickle, tqdm and the ResNet-50 model from TensorFlow are the primary dependencies.
- Two pickle files trained on the Fashion Product Images Dataset are required for the system to recommend similar images.
- CUDA installation might be required so that Tensorflow can use the dedicated GPU (CUDA can speed up the process of pickle file generation)
  
## Installation and Usage

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/IIIIIGODIIIII/Fashion_Recommendation_System.git
   cd Fashion_Recommendation_System

2. Download the required pickle files from the provided Google Drive link.
   
3. Install dependencies:

   ````python
   pip install tensorflow==2.10 scikit-learn==1.3.0 tqdm==4.66.1 numpy==1.23.5 pickle

### Usage
1. First run Generate Embeddings.ipynb to generate the pickle files. If you have already downloaded them from my google drive then go to 2nd step.
   
2. Run the jupyter notebook for the image recommendation system (Testing.ipynb). 

3. Follow the prompts to input an image and receive five recommended images from the dataset.

## Additional Information

### Documentation
For detailed information on the implementation and the theory behind the Recommendation System, refer to the Fashion Recommendation System.docx included in the repository.

### Links and Resources
1. [Fashion Product Images Dataset on Kaggle](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset)

2. Link to the Google Drive for required pickle files: [Google Drive](https://drive.google.com/drive/folders/1xyDKN-6CvGBauixo8JohUwYsMzK3HcQ7?usp=sharing)
   You might need to generate Filenames.pkl file again to make the project work
  
### Future Plans

Future updates may include:

- Optimization of recommendation algorithms.
- Support for custom image datasets.
- Integration with web or mobile applications for user-friendly interactions.
