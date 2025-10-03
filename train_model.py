# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

print("--- Starting Model Training ---")

# 1. Load Data
# The 'comment' parameter tells pandas to ignore header lines that start with '#'
try:
    df = pd.read_csv('cumulative.csv', comment='#')
except FileNotFoundError:
    print("Error: 'cumulative.csv' not found. Please download it from Kaggle and place it in the same folder.")
    exit()

# 2. Data Cleaning and Preprocessing
cols_to_use = [
    'koi_disposition', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec',
    'koi_period', 'koi_duration', 'koi_depth', 'koi_prad', 'koi_teq', 'koi_insol'
]
df = df[cols_to_use]
df = df.dropna()

# 3. Feature Engineering and Label Encoding
df['koi_disposition'] = df['koi_disposition'].apply(lambda x: 1 if x in ['CONFIRMED', 'CANDIDATE'] else 0)

# Separate features (X) from the target label (y)
X = df.drop('koi_disposition', axis=1)
y = df['koi_disposition']

# 4. Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5. Scale the Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Train the Random Forest Model
print("Training the Random Forest model...")
model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train_scaled, y_train)
print("Model training complete.")

# 7. Evaluate the Model
predictions = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy on Test Data: {accuracy * 100:.2f}%")

# 8. Save the Model and the Scaler
joblib.dump(model, 'planet_model.joblib')
joblib.dump(scaler, 'scaler.joblib')

print("--- Model and scaler saved successfully! ---")