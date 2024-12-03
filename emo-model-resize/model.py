import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Reshape, MultiHeadAttention, LayerNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
import pickle
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.mixed_precision import set_global_policy


set_global_policy('mixed_float16')

def create_model():

    inputs = Input(shape=(224, 224, 1))

    # First convo block
    x = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Second convo block
    x = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    # Third convo block (added to handle larger input)
    x = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    # Reshape for multi-head attention
    x = Reshape((-1, 128))(x) 

    # Multi-head attention
    attention_output = MultiHeadAttention(num_heads=4, key_dim=128)(x, x)  
    attention_output = LayerNormalization()(attention_output)  

    # Fully connected layers
    x = Flatten()(attention_output)
    x = Dense(256, activation='relu')(x)  
    x = Dropout(0.5)(x)
    outputs = Dense(7, activation='softmax', dtype='float32')(x) 

    model = Model(inputs=inputs, outputs=outputs)
    
    return model


def train_model(model, train_dataset, val_dataset, num_epochs, learning_rate):
    
    optimizer = Adam(learning_rate)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    checkpoint_callback = ModelCheckpoint(
        filepath='model_checkpoint.keras',
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    )
    csv_logger = CSVLogger('emo-model-resize/training_log.csv', append=True)
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=num_epochs,  # Longer training
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
    
    plt.savefig('emo-model-resize/training_history_plot.png')
    plt.close() 
    
def save_training_history(history):
    with open('emo-model-resize/training_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)


if __name__ == "__main__":
    with open("preprocessed_data_resize.pkl",'rb') as f:
        X_train, X_val, X_test, y_train, y_val, y_test = pickle.load(f)
    print("Data loaded")
    
    model = create_model()
    batch_size, num_epochs, learning_rate = 512, 50, 1e-3
    

    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1000).batch(batch_size).prefetch(AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size).prefetch(AUTOTUNE)
    
    print("Start training")
    history = train_model(model, train_dataset, val_dataset, num_epochs, learning_rate)
    
    model.save('emo-model-resize/final_model.keras')
    save_training_history(history)  
    plot_training_history(history)  # Save the plot as an image
    