import matplotlib.pyplot as plt
import pandas as pd
import pickle

def plot_training_history(history, output_path):
    # Plot accuracy
    plt.figure(figsize=(12, 5))
    
    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Save the plot as a PNG file
    plt.savefig(output_path)
    plt.close()  # Close the plot to avoid display on non-interactive environments
    
if __name__ == "__main__":
    history = pd.read_csv("training_models/stage3/emo-model-2/training_log.csv")
    output_path ="training_emo_model2.png"
    plot_training_history(history,output_path)  # Save the plot as an image

    