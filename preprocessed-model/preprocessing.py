import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'cic2020.csv'  # Adjust path as needed
data = pd.read_csv(file_path)

# ================== Data Preprocessing ==================
# Handle missing values with SimpleImputer (for numerical columns)
numerical_columns = data.select_dtypes(include=[np.number]).columns
imputer = SimpleImputer(strategy='mean')
data[numerical_columns] = imputer.fit_transform(data[numerical_columns])

# Encode categorical features
categorical_columns = ['Protocol', 'Label', 'Label.1']
label_encoders = {col: LabelEncoder().fit(data[col].astype(str)) for col in categorical_columns}

for col in categorical_columns:
    data[col] = label_encoders[col].transform(data[col].astype(str))

# Normalize the numerical feature set (excluding label columns)
features_to_scale = data.drop(columns=['Label', 'Label.1']).columns  # Exclude label columns
scaler = StandardScaler()
data[features_to_scale] = scaler.fit_transform(data[features_to_scale])

# ================== Feature Selection ==================
# Using a RandomForest classifier for feature importance
X = data.drop(columns=['Label', 'Label.1'])
y = data['Label']  # Using 'Label' as the primary label

rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)
importances = rf.feature_importances_

# Select top features based on importance
important_features = np.argsort(importances)[-10:]  # Select top 10 features
X_selected = X.iloc[:, important_features]

# ================== Data Balancing ==================
# Using SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_selected, y)

# ================== Visualization ==================
plt.figure(figsize=(15, 6))
sns.countplot(x='Protocol', data=data, palette='hls')
plt.xticks(rotation=90)
plt.show()

# Save the processed data and balanced features for future use
balanced_data = pd.DataFrame(X_balanced, columns=[X.columns[i] for i in important_features])
balanced_data['Label'] = y_balanced
balanced_data.to_csv('processed_cic_darknet2020_data.csv', index=False)  # Save balanced and processed data

print("Data preprocessing, feature selection, and balancing completed.")
