import sys
import io
from sklearn.preprocessing import LabelEncoder

# Redirect standard output to a text file
sys.stdout = open('DNN_results_cleaned2.txt', 'w', encoding='utf-8')

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
# Load your dataset (adjust the path as necessary)
df = pd.read_csv('processed_cic_darknet2020_data.csv')  # Adjust the path

# Handle categorical data: Use LabelEncoder to convert strings to numbers
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Separate features and target (assuming the last column is the target)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Normalize the features using MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build the deep neural network model
model = models.Sequential()

# Define the input layer with the correct shape using Input
model.add(layers.Input(shape=(X_train.shape[1],)))
model.add(layers.Dense(79, activation='relu'))
model.add(layers.Dropout(0.2))

# Additional hidden layers
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.2))

model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.2))

model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dropout(0.2))


model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dropout(0.2))

model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dropout(0.2))

# Output layer with softmax activation for multi-class classification
model.add(layers.Dense(11, activation='softmax'))  # Updated to 11 classes

# Compile the model with Adam optimizer and sparse categorical crossentropy loss
initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True)

model.compile(optimizer=optimizers.Adam(learning_rate=lr_schedule),
              loss='sparse_categorical_crossentropy',  # Use sparse categorical crossentropy for integer targets
              metrics=['accuracy'])

# Train the model and store the history
history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_split=0.33)

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(X_test, y_test)

# Print results to the text file
print(f'Test loss: {test_loss}')
print(f'Test accuracy: {test_acc}')

# Optionally, store training history metrics
print("\nTraining History:")
for key in history.history.keys():
    print(f"{key}: {history.history[key]}")


import matplotlib.pyplot as plt

# Assuming 'history' is the variable storing the result of model.fit()
# Example:
# history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100)

# Plotting the Loss
plt.figure(figsize=(12, 6))

# Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Show the plots
plt.tight_layout()
plt.show()
