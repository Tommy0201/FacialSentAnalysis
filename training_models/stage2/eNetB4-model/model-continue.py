import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB4  # Changed to B4 ~ 19B
import pickle
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from tensorflow.keras.mixed_precision import set_global_policy

# Enable mixed precision
set_global_policy('float32')
dir_name = "eNetB4-model/"

# Define a custom spatial attention layer
@tf.keras.utils.register_keras_serializable()
class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self, trainable=True, dtype=None, **kwargs):
        super(SpatialAttention, self).__init__(trainable=trainable, dtype=dtype, **kwargs)
        self.conv = layers.Conv2D(1, kernel_size=7, padding="same", activation="sigmoid")

    def call(self, inputs):
        if len(inputs.shape) == 2:
            inputs = tf.reshape(inputs, (-1, 1, 1, inputs.shape[-1]))
        attn_map = self.conv(inputs)
        return inputs * attn_map

    def get_config(self):
        base_config = super(SpatialAttention, self).get_config()
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)



def continue_train_model(model, train_dataset, val_dataset, learning_rate, log_file, total_epochs, curr_epochs):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    checkpoint_callback = ModelCheckpoint(
        filepath=dir_name + "model_checkpoint_eNetB4.keras",
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    )
    csv_logger = CSVLogger(log_file, append=True)
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=total_epochs,
        initial_epoch = curr_epochs,
        verbose=1,
        callbacks=[checkpoint_callback, csv_logger, early_stopping]
    )
    return history

def save_training_history(history, file_name=dir_name + "training_history_eNetB4.pkl"):
    with open(file_name, "wb") as f:
        pickle.dump(history.history, f)

def plot_training_history(history, file_name=dir_name + "training_history_plot_eNetB4.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.savefig(file_name)
    plt.close()

if __name__ == "__main__":
    with open('preprocessed_data_resize.pkl', 'rb') as f:
        X_train, X_val, X_test, y_train, y_val, y_test = pickle.load(f)
    print("Finished loading variables")

    batch_size, epoch_size, learning_rate = 1024, 50, 1e-3

    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(32000).batch(batch_size).prefetch(AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size).prefetch(AUTOTUNE)
    print("Dataset prepared")
    
    curr_epochs = 31
    add_epochs = 69
    total_epochs = curr_epochs + add_epochs
    
    model_path = dir_name+ "model_checkpoint_eNetB4.keras"
    model = tf.keras.models.load_model(model_path, custom_objects={'SpatialAttention': SpatialAttention})
    
    print("Model Loaded")

    
    history = continue_train_model(model, train_dataset, val_dataset, learning_rate, 
                                   log_file=dir_name + "training_log_eNetB4.csv", 
                                   total_epochs=total_epochs, 
                                   curr_epochs=curr_epochs)
    
    # Save the final model and training history
    model.save(dir_name + "final_model_eNetB4.keras")
    save_training_history(history)
    plot_training_history(history)
