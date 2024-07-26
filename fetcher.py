import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

df = pd.read_excel("C:\\Users\\nicro\\OneDrive\\Desktop\\New folder\\Data1.xlsx", engine="openpyxl")

print(df.head())

df = df.fillna(df.mean())


X = df.drop(columns=['Price of water (Rs./Kl)'])
y = df['Price of water (Rs./Kl)']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# Load the model for future predictions
loaded_model = tf.keras.models.load_model("water_price_model.h5")

# Load the scaler
loaded_scaler = joblib.load("scaler.pkl")

# Predict water prices on the test data using the loaded model
y_pred = loaded_model.predict(np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]])
)

# Display some predictions
print("Predicted Prices:", y_pred[:5].flatten())
print("Actual Prices:", y_test[:5].values)