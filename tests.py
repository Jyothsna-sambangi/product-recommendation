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

st.title('Product Recommendation System')

# Debug message
st.write("Starting application...")

# Load the precomputed feature list and filenames
try:
    with open('embeddings2.pkl', 'rb') as f:
        feature_list = np.array(pickle.load(f))
    st.write("Loaded embeddings2.pkl successfully.")
except Exception as e:
    st.write(f"Error loading embeddings2.pkl: {e}")

try:
    with open('filenames2.pkl', 'rb') as f:
        filenames = pickle.load(f)
    st.write("Loaded filenames2.pkl successfully.")
except Exception as e:
    st.write(f"Error loading filenames2.pkl: {e}")

# Initialize the ResNet50 model
try:
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    model = tf.keras.Sequential([
        base_model,
        GlobalMaxPooling2D()
    ])
    st.write("ResNet50 model initialized successfully.")
except Exception as e:
    st.write(f"Error initializing ResNet50 model: {e}")

# Function to save uploaded file
def save_uploaded_file(uploaded_file):
    try:
        if not os.path.exists('uploads'):
            os.makedirs('uploads')
        file_path = os.path.join('uploads', uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        st.write(f"File saved successfully at {file_path}.")
        return file_path
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

# Function to extract features from an image
def extract_image_features(img_path, model):
    try:
        img = keras_image.load_img(img_path, target_size=(224, 224))
        img_array = keras_image.img_to_array(img)
        img_array_expanded = np.expand_dims(img_array, axis=0)
        img_preprocessed = preprocess_input(img_array_expanded)
        features = model.predict(img_preprocessed).flatten()
        st.write("Image features extracted successfully.")
        return features / norm(features)
    except Exception as e:
        st.write(f"Error extracting image features: {e}")
        return None

# Function to find similar images
def find_similar_images(features, feature_list):
    try:
        knn = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
        knn.fit(feature_list)
        _, indices = knn.kneighbors([features])
        st.write("Similar images found successfully.")
        return indices
    except Exception as e:
        st.write(f"Error finding similar images: {e}")
        return None

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
        if features is not None:
            indices = find_similar_images(features, feature_list)
            if indices is not None:
                # Display recommended images
                st.write("Here are some similar images:")
                columns = st.columns(5)
                for i, col in enumerate(columns):
                    col.image(filenames[indices[0][i]])
            else:
                st.error("An error occurred while finding similar images.")
        else:
            st.error("An error occurred while extracting image features.")
    else:
        st.error("An error occurred during file upload.")
