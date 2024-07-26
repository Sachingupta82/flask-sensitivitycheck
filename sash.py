import shap
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Load the dataset
df = pd.read_excel("C:\\Users\\nicro\\OneDrive\\Desktop\\New folder\\Data1.xlsx", engine="openpyxl")
df = df.fillna(df.mean())

X = df.drop(columns=['Price of water (Rs./Kl)'])
y = df['Price of water (Rs./Kl)']

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Load the model
loaded_model = tf.keras.models.load_model("water_price_model.h5")

# SHAP Explainer
explainer = shap.KernelExplainer(loaded_model.predict, X_train)
shap_values = explainer.shap_values(X_test)

# Compute mean SHAP values for each feature
mean_shap_values = np.mean(np.abs(shap_values), axis=0)

# Create a DataFrame to show feature sensitivities
sensitivity_df = pd.DataFrame(mean_shap_values, columns=X.columns, index=['Sensitivity'])
print(sensitivity_df)
