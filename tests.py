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

# Load the precomputed feature list and filenames
with open('embeddings2.pkl', 'rb') as f:
    feature_list = np.array(pickle.load(f))

with open('filenames2.pkl', 'rb') as f:
    filenames = pickle.load(f)

# Initialize the ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
model = tf.keras.Sequential([
    base_model,
    GlobalMaxPooling2D()
])

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

# Function to convert GitHub path to local path format
def github_to_local_path(github_path):
    return github_path.replace("/", "\\")

# Main Streamlit app code
def main():
    st.title("Product Recommendation System")

    # Sidebar for uploading or displaying image from GitHub
    st.sidebar.header("Image Options")
    option = st.sidebar.radio(
        "Choose an option:",
        ("Upload an image", "Display image from GitHub")
    )

    if option == "Upload an image":
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
                    image_path = filenames[indices[0][i]]
                    col.image(image_path)

            else:
                st.error("An error occurred during file upload.")

    elif option == "Display image from GitHub":
        # Example GitHub path (replace with your actual GitHub paths)
        github_image_path = "images_with_product_ids/10037.jpg"

        # Convert GitHub path to local path format (if needed)
        local_image_path = github_to_local_path(github_image_path)

        # Display using converted local path
        if os.path.exists(local_image_path):
            st.image(local_image_path, caption='Image from GitHub (Converted Local Path)')
        else:
            st.error(f"Image not found at path: {local_image_path}")

        # Optionally, display directly using GitHub path if your environment supports it
        # st.image(github_image_path, caption='Image from GitHub (GitHub Path)')

        # Optionally, normalize the path to handle mixed format
        normalized_path = os.path.normpath(github_image_path)
        st.image(normalized_path, caption='Normalized Image Path')

if __name__ == "__main__":
    main()
