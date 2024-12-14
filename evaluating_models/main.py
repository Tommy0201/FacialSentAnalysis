import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pickle

# Define emotion mapping
emotion_dict = {
    1: "Surprise",
    2: "Fear",
    3: "Disgust",
    4: "Happy",
    5: "Sad",
    6: "Angry",
    7: "Neutral"
}

def load_model_and_data(model_path, test_data_path):
    """Load the trained model and test data"""
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Load test data
    with open(test_data_path, 'rb') as f:
        _, _, X_test, _, _, y_test = pickle.load(f)
    
    # return model, X_test, y_test
    X_test_resized = tf.image.resize(X_test, [224, 224])
    
    return model, X_test_resized.numpy(), y_test

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance and return metrics"""
    # Convert to dataset
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(512)
    
    # Get overall metrics
    loss, accuracy = model.evaluate(test_dataset)
    
    # Get predictions
    y_pred = model.predict(test_dataset)
    # Add 1 to predictions since our classes are 1-7 instead of 0-6
    y_pred_classes = np.argmax(y_pred, axis=1) + 1
    y_test_classes = np.argmax(y_test, axis=1) + 1
    
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
    class_names = [emotion_dict[i] for i in range(1, 8)]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def plot_sample_predictions(X_test, y_true, y_pred, num_samples=5):
    """Plot sample images with their predictions"""
    indices = np.random.randint(0, len(X_test), num_samples)
    
    plt.figure(figsize=(15, 3))
    for i, idx in enumerate(indices):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(X_test[idx], cmap='gray')
        true_label = emotion_dict[y_true[idx]]
        pred_label = emotion_dict[y_pred[idx]]
        plt.title(f'True: {true_label}\nPred: {pred_label}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('sample_predictions.png')
    plt.close()

def main():
    # Load model and data
    model, X_test, y_test = load_model_and_data(
        'training_models/stage3/eNetB4-model2/model_checkpoint_eNetB4.keras',
        'training_models/preprocessed2_data.pkl'
    )
    
    # Evaluate model
    print("Evaluating model...")
    results = evaluate_model(model, X_test, y_test)
    
    # Print metrics
    print(f"\nTest Loss: {results['loss']:.4f}")
    print(f"Test Accuracy: {results['accuracy']:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    class_names = [emotion_dict[i] for i in range(1, 8)]
    print(classification_report(results['y_test_classes'], 
                              results['y_pred_classes'],
                              target_names=class_names))
    
    # Plot confusion matrix
    print("\nGenerating confusion matrix...")
    plot_confusion_matrix(results['y_test_classes'], 
                         results['y_pred_classes'])
    
    # Plot sample predictions
    print("Generating sample predictions visualization...")
    plot_sample_predictions(X_test,
                          results['y_test_classes'],
                          results['y_pred_classes'])
    
    # Calculate per-class accuracy
    print("\nPer-class accuracy:")
    cm = confusion_matrix(results['y_test_classes'], results['y_pred_classes'])
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    for i, accuracy in enumerate(per_class_accuracy, 1):
        emotion = emotion_dict[i]
        print(f"{emotion}: {accuracy:.4f}")

if __name__ == "__main__":
    main()