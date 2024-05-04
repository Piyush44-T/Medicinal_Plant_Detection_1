import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
main_model = load_model('Model1.keras')  

# List of medicinal plant names
out = ['Aloevera', 'Amla', 'Amruthaballi', 'Arali', 'Astma_weed', 'Badipala', 'Balloon_Vine', 'Bamboo', 'Beans',
       'Betel', 'Bhrami', 'Bringaraja', 'Caricature', 'Castor', 'Catharanthus', 'Chakte', 'Chilly',
       'Citron lime (herelikai)', 'Coffee', 'Common rue(naagdalli)', 'Coriender', 'Curry', 'Doddpathre', 'Drumstick',
       'Ekka', 'Eucalyptus', 'Ganigale', 'Ganike', 'Gasagase', 'Ginger', 'Globe Amarnath', 'Guava', 'Henna', 'Hibiscus',
       'Honge', 'Insulin', 'Jackfruit', 'Jasmine', 'Kambajala', 'Kasambruga', 'Kohlrabi', 'Lantana', 'Lemon',
       'Lemongrass', 'Malabar_Nut', 'Malabar_Spinach', 'Mango', 'Marigold', 'Mint', 'Neem', 'Nelavembu', 'Nerale',
       'Nooni', 'Onion', 'Padri', 'Palak(Spinach)', 'Papaya', 'Parijatha', 'Pea', 'Pepper', 'Pomoegranate', 'Pumpkin',
       'Raddish', 'Rose', 'Sampige', 'Sapota', 'Seethaashoka', 'Seethapala', 'Spinach1', 'Tamarind', 'Taro', 'Tecoma',
       'Thumbe', 'Tomato', 'Tulsi', 'Turmeric', 'ashoka', 'augmented_images', 'camphor', 'kamakasturi', 'kepala']


def preprocess_image(image):
    # Resize image to match model input size and normalize pixel values
    img = image.resize((224, 224))  # Resize image
    img_array = np.array(img) / 255.  # Convert image to numpy array and normalize pixel values
    return img_array


def predict_plant_species(image):
    # Preprocess the image
    img = preprocess_image(image)
    img = img.reshape(1, 224, 224, 3)

    # Make prediction
    res = main_model.predict(img)
    predicted_class = out[np.argmax(res)]
    return predicted_class


st.title("Medicinal Plant Species Prediction")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    image = Image.open(uploaded_file)

    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Make prediction
    predicted_species = predict_plant_species(image)
    st.write(f'Predicted Plant Species: {predicted_species}')
