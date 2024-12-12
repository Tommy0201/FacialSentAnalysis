import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from PIL import Image
import io
import pickle 

def summarize_data(data):
    # General Information
    print("Dataset Information:")
    print(data.info())
    
    # Check for Missing Values
    print("\nMissing Values:")
    print(data.isnull().sum())
    
    # Check Class Distribution
    print("\nClass Distribution:")
    print(data['label'].value_counts())
    
    # Check Unique Sources
    print("\nSource Distribution:")
    print(data['source'].value_counts())

    # Display Sample Rows
    print("\nSample Data:")
    print(data.head())

    # Calculate and Display Total Samples
    print(f"\nTotal Samples: {len(data)}")

def decode_image(image_data, target_size=(48,48)):
    image_bytes = eval(image_data)['bytes']  
    img = Image.open(io.BytesIO(image_bytes)).convert('L')
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    return img_array

def preprocess_data(data):
    images = np.array([decode_image(img) for img in data['image']])
    images = images[...,np.newaxis]
    labels = to_categorical(data['label']-1,num_classes=7)
    
    #train and test + validation
    X_train, X_temp, y_train, y_temp = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    #train and validation
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    data = pd.read_csv("training_models/data/affectnet.csv")
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(data)
    # Save the variables
    print(f"Train data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    with open('training_models/affectnet_data.pkl', 'wb') as f:
        pickle.dump((X_train, X_val, X_test, y_train, y_val, y_test), f)
    print("Variables saved.")

    # summarize_data(data)  