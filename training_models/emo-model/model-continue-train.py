import tensorflow as tf
import pickle
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
import matplotlib.pyplot as plt
from tensorflow.keras.mixed_precision import set_global_policy

# Enable mixed precision
set_global_policy('mixed_float16')

def continue_training(model, train_dataset, val_dataset, initial_epochs=3, new_epochs=5):

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Callbacks for continuing training
    checkpoint_callback = ModelCheckpoint(
        filepath='model_checkpoint_continued.keras',
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    )
    csv_logger = CSVLogger('training_log_continued.csv', append=True)
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

    # Continue training
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=new_epochs,
        initial_epoch=initial_epochs,  # Resume from where it left off
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
    # Load preprocessed data
    with open('preprocessed_data.pkl', 'rb') as f:
        X_train, X_val, X_test, y_train, y_val, y_test = pickle.load(f)

    # Prepare datasets
    batch_size = 256
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1000).batch(batch_size).prefetch(AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size).prefetch(AUTOTUNE)

    # Continue training
    initial_epochs = 12  # Assuming the initial training was for 12 epochs
    additional_epochs = 38  # Number of additional epochs to train
    total_epochs = initial_epochs + additional_epochs
    
    model_path = 'model_checkpoint.keras'  # Path to the saved model
    model = tf.keras.models.load_model(model_path)
    
    print("Model loaded")
    
    history = continue_training(model, train_dataset, val_dataset, initial_epochs=initial_epochs, new_epochs=total_epochs)
    # Save the final model and updated history
    model.save('final_model_updated.h5')
    save_training_history(history)  # Update training history
    plot_training_history(history)  # Save updated training plot
