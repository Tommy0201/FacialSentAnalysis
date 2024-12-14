import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pickle
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB4, ResNet50
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Reshape, MultiHeadAttention, LayerNormalization

import os
import json

# Create results directory relative to script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, 's3-cus-emo')  # Example directory name
os.makedirs(RESULTS_DIR, exist_ok=True)



class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = layers.Conv2D(1, kernel_size=7, padding="same", activation="sigmoid")

    def call(self, inputs):
        if len(inputs.shape) == 2:
            inputs = tf.reshape(inputs,(-1, 1, 1, inputs.shape[-1]))
        attn_map = self.conv(inputs)
        return inputs * attn_map


# Define emotion mapping
# emotion_dict = {
#     1: "Surprise",
#     2: "Fear",
#     3: "Disgust",
#     4: "Happy",
#     5: "Sad",
#     6: "Angry",
#     7: "Neutral"
# }

emotion_dict = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprise",
}

def create_eNetB0_model(num_classes=7):
    base_model = EfficientNetB0(include_top=False, weights="imagenet", pooling="avg", input_shape=(48,48,3))
    base_model.trainable = False  # Freeze the base model weights

    inputs = tf.keras.Input(shape=(48,48,1))
    x = layers.Concatenate()([inputs, inputs, inputs])
    x = base_model(x, training=False)
    x = SpatialAttention()(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype='float32')(x)

    model = Model(inputs, outputs)
    return model

def create_eNetB4_model(num_classes=7):
    base_model = EfficientNetB4(include_top=False, weights="imagenet", pooling="avg", input_shape=(224,224,3))
    base_model.trainable = False  

    inputs = tf.keras.Input(shape=(224,224,1)) # Resize image since large parameters
    x = layers.Concatenate()([inputs, inputs, inputs])
    x = base_model(x, training=False)
    x = SpatialAttention()(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype='float32')(x)

    model = Model(inputs, outputs)
    return model

def create_resNet_model(num_classes=7):
    # Load ResNet-50 without the top layers
    base_model = ResNet50(include_top=False, weights="imagenet", pooling=None, input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze the base model weights

    inputs = tf.keras.Input(shape=(224, 224, 1))    
    x = layers.Concatenate()([inputs, inputs, inputs])
    x = base_model(x, training=False)    
    x = SpatialAttention()(x)    
    x = layers.GlobalAveragePooling2D()(x)    
    outputs = layers.Dense(num_classes, activation="softmax", dtype='float32')(x)

    model = Model(inputs, outputs)
    return model

def load_eNetB0_model_and_data(model_path, test_data_path):
    model = create_eNetB0_model()
    model.load_weights(model_path)
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    with open(test_data_path, 'rb') as f:
        _, _, X_test, _, _, y_test = pickle.load(f)
            
    return model, X_test, y_test

def load_eNetB4_model_and_data(model_path, test_data_path):
    """Load the trained model and test data"""
    # Create the model first, then load weights
    model = create_eNetB4_model()
    model.load_weights(model_path)
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    with open(test_data_path, 'rb') as f:
        _, _, X_test, _, _, y_test = pickle.load(f)
    
    # Resize images from 48x48 to 224x224
    X_test = tf.image.resize(X_test, [224, 224]).numpy()
            
    return model, X_test, y_test

def load_resNet_model_and_data(model_path, test_data_path):
    """Load the trained model and test data"""
    # Create the model first, then load weights
    model = create_resNet_model()
    model.load_weights(model_path)
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    with open(test_data_path, 'rb') as f:
        _, _, X_test, _, _, y_test = pickle.load(f)
    
    # Resize images from 48x48 to 224x224
    X_test = tf.image.resize(X_test, [224, 224]).numpy()
            
    return model, X_test, y_test



def load_customize_model_and_data(model_path, test_data_path):
    """Load the trained model and test data"""
    # Load model with custom objects
    model = tf.keras.models.load_model(model_path)
    
    # Print model input shape for debugging
    print("Expected input shape:", model.input_shape)
    
    # Load test data
    with open(test_data_path, 'rb') as f:
        _, _, X_test, _, _, y_test = pickle.load(f)
    
    print("Input data shape:", X_test.shape)
    
    return model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance and return metrics"""
    # Convert to dataset
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(512)
    
    # Get overall metrics
    loss, accuracy = model.evaluate(test_dataset)
    
    # Get predictions
    y_pred = model.predict(test_dataset)
    # Add 1 to predictions since our classes are 1-7 instead of 0-6
    y_pred_classes = np.argmax(y_pred, axis=1) 
    y_test_classes = np.argmax(y_test, axis=1) 
    
    return {
        'loss': loss,
        'accuracy': accuracy,
        'y_pred': y_pred,
        'y_pred_classes': y_pred_classes,
        'y_test_classes': y_test_classes
    }

def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    class_names = [emotion_dict[i] for i in range(0, 7)]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
    plt.savefig(save_path)
    plt.close()

def plot_sample_predictions(X_test, y_true, y_pred, num_samples=5):
    """Plot sample images with their predictions"""
    indices = np.random.randint(0, len(X_test), num_samples)
    
    plt.figure(figsize=(15, 3))
    for i, idx in enumerate(indices):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(X_test[idx, :, :, 0], cmap='gray')
        true_label = emotion_dict[y_true[idx]]
        pred_label = emotion_dict[y_pred[idx]]
        plt.title(f'True: {true_label}\nPred: {pred_label}')
        plt.axis('off')
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, 'sample_predictions.png')
    plt.savefig(save_path)
    plt.close()

def main():
    model, X_test, y_test = load_customize_model_and_data(
        'training_models/stage3/emo-model-2/final_model_2.keras',
        'training_models/preprocessed2_data.pkl'
    )
    
    print("Evaluating model...")
    results = evaluate_model(model, X_test, y_test)
    
    metrics_dict = {
        'test_loss': float(results['loss']),
        'test_accuracy': float(results['accuracy']),
        'classification_report': classification_report(
            results['y_test_classes'],
            results['y_pred_classes'],
            target_names=[emotion_dict[i] for i in range(0, 7)],
            output_dict=True
        ),
        'per_class_accuracy': {}
    }
    
    # Print and store results
    print(f"\nTest Loss: {results['loss']:.4f}")
    print(f"Test Accuracy: {results['accuracy']:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(
        results['y_test_classes'],
        results['y_pred_classes'],
        target_names=[emotion_dict[i] for i in range(0, 7)]
    ))
    
    # Generate visualizations
    print("\nGenerating confusion matrix...")
    plot_confusion_matrix(results['y_test_classes'], 
                         results['y_pred_classes'])
    
    print("Generating sample predictions visualization...")
    plot_sample_predictions(X_test,
                          results['y_test_classes'],
                          results['y_pred_classes'])
    
    # Calculate and save per-class accuracy
    print("\nPer-class accuracy:")
    cm = confusion_matrix(results['y_test_classes'], results['y_pred_classes'])
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    for i, accuracy in enumerate(per_class_accuracy):
        emotion = emotion_dict[i]
        metrics_dict['per_class_accuracy'][emotion] = float(accuracy)
        print(f"{emotion}: {accuracy:.4f}")
    
    # Save metrics to JSON
    metrics_path = os.path.join(RESULTS_DIR, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_dict, f, indent=4)
    
    print(f"\nAll results have been saved to: {RESULTS_DIR}")

if __name__ == "__main__":
    main()