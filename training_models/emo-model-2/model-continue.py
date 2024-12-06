import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Reshape, MultiHeadAttention, LayerNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
import pickle
from tensorflow.keras.mixed_precision import set_global_policy

# Enable mixed precision
set_global_policy('float32')
dir_name = "emo-model-2/"


def continue_training(model, train_dataset, val_dataset, learning_rate, total_epochs, curr_epochs):
    model.compile(
        optimizer=Adam(learning_rate), 
        loss='categorical_crossentropy', 
        metrics=['accuracy'])
    model.summary()
    checkpoint_callback = ModelCheckpoint(
        filepath='model_checkpoint.keras',
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    )
    csv_logger = CSVLogger(dir_name+'training_log.csv', append=True)
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True)
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=total_epochs,
        initial_epoch = curr_epochs,
        verbose=1,
        callbacks=[checkpoint_callback, csv_logger, early_stopping]
    )
    return history


def plot_training_history(history):
    # Plot accuracy
    plt.figure(figsize=(12, 5))
    
    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Save the plot as a PNG file
    plt.savefig(dir_name+'training_history_plot.png')
    plt.close()  # Close the plot to avoid display on non-interactive environments
def save_training_history(history):
    # Save the training history to a pickle file
    with open(dir_name+'training_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)
    print("Training history saved to 'training_history.pkl'.")

    
if __name__ == "__main__":
    with open('preprocessed2_data.pkl', 'rb') as f:
        X_train, X_val, X_test, y_train, y_val, y_test = pickle.load(f)
    print("Finish loading variables")
    
    # Prepare datasets
    batch_size = 256
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(32000).batch(batch_size).prefetch(AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size).prefetch(AUTOTUNE)
    
    curr_epochs = 100
    add_epochs = 200
    total_epochs = curr_epochs+add_epochs
    learning_rate = 1e-4
    
    model_path = dir_name + ""
    model = tf.keras.models.load_model(model_path)
    history = continue_training(model, train_dataset, val_dataset, learning_rate,
                                total_epochs=total_epochs,
                                curr_epochs=curr_epochs)
    
    # Save the final model and history
    model.save(dir_name+'final_model.keras')
    save_training_history(history)  # Save the history to a pickle file
    plot_training_history(history)  # Save the plot as an image
