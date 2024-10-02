import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import regularizers

# Force UTF-8 encoding for standard output
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Optional: Redirect output to a log file to avoid encoding issues
sys.stdout = open('training_log.txt', 'w', encoding='utf-8')

# Load the dataset
file_path = 'data/flowdata11.binetflow.csv'  # Update this to the correct path
data = pd.read_csv(file_path)

# Handle missing values with SimpleImputer (for 'stos' and 'dtos')
imputer = SimpleImputer(strategy='mean')
data[['stos', 'dtos']] = imputer.fit_transform(data[['stos', 'dtos']])

# Encode categorical features
label_encoders = {}
categorical_columns = ['proto', 'dir', 'state', 'label']

for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le

# Prepare features (X) and target (y)
X = data.drop(columns=['label', 'Family'])  # 'Family' and 'label' are not used as features
y = data['label']

# Normalize the feature set
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Determine the number of unique classes in the target variable
num_classes = len(set(y_train))  # Get the number of unique classes from the training labels

# Build the DNN model with regularization (Dropout and L2)
model = models.Sequential()

# Input layer
model.add(layers.InputLayer(input_shape=(X_train.shape[1],)))

# Hidden layers with Dropout and L2 regularization
model.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)))  # L2 regularization
model.add(layers.Dropout(0.5))  # Dropout layer to prevent overfitting

model.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))  # L2 regularization
model.add(layers.Dropout(0.5))  # Dropout layer to prevent overfitting

model.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))  # L2 regularization

# Output layer (for multi-class classification)
model.add(layers.Dense(num_classes, activation='softmax'))

# Compile the model with optimizer
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_split=0.2)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')