from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Reshape, MultiHeadAttention, LayerNormalization
)

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
outputs = Dense(7, activation='softmax')(x) 

model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
