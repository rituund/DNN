import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Load your dataset (adjust the path as necessary)
df = pd.read_csv('cic2020.csv')  # Adjust the path

# Separate features and target (assuming the last column is the target)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# One-hot encode any categorical features in X and the target variable y
X_encoded = pd.get_dummies(X)  # This will one-hot encode all categorical features in X
encoder = OneHotEncoder(sparse_output=False)  # Use sparse_output=False for newer versions
y_encoded = encoder.fit_transform(y.values.reshape(-1, 1))

# Normalize the features using MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Build the deep neural network model
model = models.Sequential()

# Define the input layer with the correct shape using Input
model.add(layers.Input(shape=(X_train.shape[1],)))  # Updated input definition
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.2))

# Additional hidden layers
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.2))

model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.2))

model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.2))

model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.2))

# Output layer with the correct number of classes (11 instead of 4)
model.add(layers.Dense(11, activation='softmax'))  # Updated to 11 classes

# Compile the model
initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True)

model.compile(optimizer=optimizers.Adam(learning_rate=lr_schedule),
              loss='categorical_crossentropy',  # Use categorical crossentropy for one-hot encoded target
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_split=0.33)

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')
