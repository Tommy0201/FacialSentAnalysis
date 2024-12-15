import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential
from tensorflow.keras.applications import EfficientNetB0
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
class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = layers.Conv2D(1, kernel_size=7, padding="same", activation="sigmoid")

    def call(self, inputs):
        if len(inputs.shape) ==2:
            inputs = tf.reshape(inputs,(-1, 1, 1, inputs.shape[-1]))
        attn_map = self.conv(inputs)
        return inputs * attn_map
    

# Define the EfficientNet model with attention
def create_model(num_classes=7):
    base_model = EfficientNetB4(include_top=False, weights="imagenet", pooling="avg", input_shape=(224,224,3))
    base_model.trainable = False  # Freeze the base model weights

    inputs = tf.keras.Input(shape=(224,224,1))
    x = layers.Concatenate()([inputs, inputs, inputs])
    x = base_model(x, training=False)
    x = SpatialAttention()(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype='float32')(x)

    model = Model(inputs, outputs)
    return model

def train_model(model, train_dataset, val_dataset, num_epochs, learning_rate, log_file):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    checkpoint_callback = ModelCheckpoint(
        filepath=dir_name+"model_checkpoint_eNet.keras",
        save_best_only = True,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    )
    csv_logger = CSVLogger(log_file, append=True)
    early_stopping = EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True)

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=num_epochs,
        verbose=1,
        callbacks = [checkpoint_callback, csv_logger, early_stopping]
    )
    return history

def save_training_history(history, file_name=dir_name+"training_history_eNet.pkl"):
    with open(file_name, "wb") as f:
        pickle.dump(history.history, f)

def plot_training_history(history, file_name=dir_name+"training_history_plot_eNet.png"):
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
    print("Finish loading variables")
    
    #MODIFY THIS BATCHSIZE
    batch_size, epoch_size, learning_rate = 512, 50, 1e-3
    
    
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(32000).batch(batch_size).prefetch(AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size).prefetch(AUTOTUNE)
    
    print("Dataset prepared")
    model = create_model(num_classes=7)
    print("Created the architecture")

    # Train the model
    print("Dataset prepared. Start training")
    history = train_model(model, train_dataset, val_dataset, num_epochs=epoch_size, learning_rate=learning_rate, log_file=dir_name+"training_log_eNet.csv")

    # Save the final model and training history
    model.save(dir_name+"final_model_eNet.keras")
    save_training_history(history)
    plot_training_history(history)

