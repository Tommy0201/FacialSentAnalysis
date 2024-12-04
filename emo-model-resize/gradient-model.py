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


# def train_model(model, train_dataset, val_dataset, num_epochs, learning_rate, accumulation_steps):
    
#     base_optimizer = Adam(learning_rate)
#     optimizer = GradientAccumulator(base_optimizer, accumulation_steps)

#     model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
#     model.summary()
#     checkpoint_callback = ModelCheckpoint(
#         filepath='model_checkpoint.keras',
#         save_best_only=True,
#         monitor='val_accuracy',
#         mode='max',
#         verbose=1
#     )
#     csv_logger = CSVLogger('emo-model-resize/training_log.csv', append=True)
#     early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    
#     history = model.fit(
#         train_dataset,
#         validation_data=val_dataset,
#         epochs=num_epochs,  # Longer training
#         verbose=2,
#         callbacks=[checkpoint_callback, csv_logger, early_stopping]
#     )
#     return history

def train_model(model, train_dataset, val_dataset, num_epochs, learning_rate, accumulation_steps=4):
    optimizer = Adam(learning_rate)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    
    # Metrics
    train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
    val_acc_metric = tf.keras.metrics.CategoricalAccuracy()
    
    # Initialize callbacks
    checkpoint_callback = ModelCheckpoint(
        filepath='model_checkpoint.keras',
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    )
    csv_logger = CSVLogger('emo-model-resize/training_log.csv', append=True)
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    callbacks = [checkpoint_callback, csv_logger, early_stopping]
    
    # Initialize callback states
    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    best_val_accuracy = 0
    patience_counter = 0
    
    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss_value = loss_fn(y, logits)
        
        grads = tape.gradient(loss_value, model.trainable_weights)
        if not hasattr(train_step, 'accumulated_gradients'):
            train_step.accumulated_gradients = [tf.zeros_like(g) for g in grads]
        
        for i, g in enumerate(grads):
            train_step.accumulated_gradients[i] += g / accumulation_steps
            
        if train_step.gradient_accumulation_count == accumulation_steps:
            optimizer.apply_gradients(zip(train_step.accumulated_gradients, model.trainable_weights))
            train_step.accumulated_gradients = [tf.zeros_like(g) for g in grads]
            train_step.gradient_accumulation_count = 0
        else:
            train_step.gradient_accumulation_count += 1
            
        return loss_value, logits
    
    @tf.function
    def val_step(x, y):
        val_logits = model(x, training=False)
        val_loss = loss_fn(y, val_logits)
        return val_loss, val_logits
    
    # Initialize gradient accumulation count
    train_step.gradient_accumulation_count = 0
    model.summary()
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training loop
        train_losses = []
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            loss_value, logits = train_step(x_batch_train, y_batch_train)
            train_acc_metric.update_state(y_batch_train, logits)
            train_losses.append(float(loss_value))
            
            if step % 50 == 0:
                print(f"Step {step}: loss = {float(loss_value):.4f}, accuracy = {float(train_acc_metric.result()):.4f}")
        
        # Validation loop
        val_losses = []
        for x_batch_val, y_batch_val in val_dataset:
            val_loss, val_logits = val_step(x_batch_val, y_batch_val)
            val_acc_metric.update_state(y_batch_val, val_logits)
            val_losses.append(float(val_loss))
        
        # Get epoch metrics
        train_acc = float(train_acc_metric.result())
        val_acc = float(val_acc_metric.result())
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        
        # Update history
        history['loss'].append(train_loss)
        history['accuracy'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        
        # Print metrics
        print(f"\nEpoch {epoch+1} results:")
        print(f"Training loss: {train_loss:.4f}, accuracy: {train_acc:.4f}")
        print(f"Validation loss: {val_loss:.4f}, accuracy: {val_acc:.4f}")
        
        # Handle callbacks
        csv_logger.on_epoch_end(epoch, {
            'loss': train_loss,
            'accuracy': train_acc,
            'val_loss': val_loss,
            'val_accuracy': val_acc
        })
        
        # ModelCheckpoint logic
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            model.save('model_checkpoint.keras')
            print("Model checkpoint saved")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # EarlyStopping logic
        if patience_counter >= early_stopping.patience:
            print(f"\nEarly stopping triggered after epoch {epoch+1}")
            break
        
        # Reset metrics
        train_acc_metric.reset_states()
        val_acc_metric.reset_states()
    
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
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print('GPU memory growth enabled')
    
    model = create_model()
    BATCH_SIZE = 128  
    ACCUMULATION_STEPS = 4  
    EFFECTIVE_BATCH_SIZE = BATCH_SIZE * ACCUMULATION_STEPS  # = 512
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-3


    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1600).batch(BATCH_SIZE).prefetch(AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE).prefetch(AUTOTUNE)
    
    print("Start training")
    history = train_model(model, train_dataset, val_dataset, NUM_EPOCHS, LEARNING_RATE, ACCUMULATION_STEPS)
    
    model.save('emo-model-resize/final_model.keras')
    save_training_history(history)  
    plot_training_history(history) 
    