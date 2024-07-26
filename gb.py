import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
import joblib

# Load the dataset
df = pd.read_excel("C:\\Users\\nicro\\OneDrive\\Desktop\\New folder\\Data1.xlsx", engine="openpyxl")
df = df.fillna(df.mean())

# Extract features and target variable
X = df.drop(columns=['Price of water (Rs./Kl)'])
y = df['Price of water (Rs./Kl)']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize and train the Gradient Boosting model
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


feature_importances = model.feature_importances_


importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
})

importance_df = importance_df.sort_values(by='Importance', ascending=False)


print("Predicted Prices:", y_pred[:5])
print("Actual Prices:", y_test[:5].values)
print("\nFeature Importances:")
print(importance_df)


joblib.dump(model, "water_price_model_gb.pkl")
joblib.dump(scaler, "scaler.pkl")
