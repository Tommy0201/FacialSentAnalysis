import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Reshape, MultiHeadAttention, LayerNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
import pickle
from tensorflow.keras.mixed_precision import set_global_policy

# Enable mixed precision
set_global_policy('mixed_float16')


def create_model():
    # Input layer
    inputs = Input(shape=(48, 48, 1))

    # First convo block
    x = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Second convo block
    x = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    # Reshape for multi-head attention
    x = Reshape((-1, 64))(x)  

    # Multi-head attention
    attention_output = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)  # Query = Key = Value = x
    attention_output = LayerNormalization()(attention_output)  

    # Fully connected layers
    x = Flatten()(attention_output)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(7, activation='softmax', dtype='float32')(x) 

    model = Model(inputs=inputs, outputs=outputs)
    
    return model

def train_model(model, train_dataset, val_dataset):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    checkpoint_callback = ModelCheckpoint(
        filepath='model_checkpoint.keras',
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    )
    csv_logger = CSVLogger('training_log.csv', append=True)
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=12,  # Longer training
        verbose=2,
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
    plt.savefig('training_history_plot.png')
    plt.close()  # Close the plot to avoid display on non-interactive environments
def save_training_history(history):
    # Save the training history to a pickle file
    with open('training_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)
    print("Training history saved to 'training_history.pkl'.")

    
if __name__ == "__main__":
    with open('preprocessed_data.pkl', 'rb') as f:
        X_train, X_val, X_test, y_train, y_val, y_test = pickle.load(f)
    print("Finish loading variables")
    model = create_model()
    print("Created the architecture")
    
    # Prepare datasets
    batch_size = 256
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1000).batch(batch_size).prefetch(AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size).prefetch(AUTOTUNE)
    
    print("Dataset prepared. Start training")
    history = train_model(model, train_dataset, val_dataset)
    
    # Save the final model and history
    model.save('final_model.h5')
    save_training_history(history)  # Save the history to a pickle file
    plot_training_history(history)  # Save the plot as an image
