import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
from github import Github

# GitHub repository and directory information
github_access_token = 'your_access_token_here'  # Replace with your GitHub access token
repository_name = 'Jyothsna-sambangi/product-recommendation'
directory_path = 'images_with_product_ids'

# Initialize GitHub instance
g = Github(github_access_token)
repo = g.get_repo(repository_name)
contents = repo.get_contents(directory_path)

# Extract filenames from GitHub
filenames = [file.name for file in contents if file.type == 'file']

# Load the precomputed feature list and filenames
with open('embeddings2.pkl', 'rb') as f:
    feature_list = np.array(pickle.load(f))

# Initialize the ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
model = tf.keras.Sequential([
    base_model,
    GlobalMaxPooling2D()
])

st.title('Product Recommendation System')

# Function to save uploaded file
def save_uploaded_file(uploaded_file):
    try:
        if not os.path.exists('uploads'):
            os.makedirs('uploads')
        file_path = os.path.join('uploads', uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

# Function to extract features from an image
def extract_image_features(img_path, model):
    img = keras_image.load_img(img_path, target_size=(224, 224))
    img_array = keras_image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_array_expanded)
    features = model.predict(img_preprocessed).flatten()
    return features / norm(features)

# Function to find similar images
def find_similar_images(features, feature_list):
    knn = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    knn.fit(feature_list)
    _, indices = knn.kneighbors([features])
    return indices

# Handle file upload
uploaded_file = st.file_uploader("Upload an image")
if uploaded_file:
    file_path = save_uploaded_file(uploaded_file)
    if file_path:
        # Display the uploaded image
        display_img = Image.open(uploaded_file)
        st.image(display_img, caption='Uploaded Image')

        # Extract features and find recommendations
        features = extract_image_features(file_path, model)
        indices = find_similar_images(features, feature_list)

        # Display recommended images
        st.write("Here are some similar images:")
        columns = st.columns(5)
        for i, col in enumerate(columns):
            image_index = indices[0][i]
            if image_index < len(filenames):
                image_filename = filenames[image_index]
                image_path = f"https://raw.githubusercontent.com/Jyothsna-sambangi/product-recommendation/main/images_with_product_ids/{image_filename}"
                col.image(image_path)
            else:
                st.warning(f"No image found for index {image_index}")
    else:
        st.error("An error occurred during file upload.")

