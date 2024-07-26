import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
import joblib

# Load the dataset
df = pd.read_excel("C:\\Users\\nicro\\OneDrive\\Desktop\\New folder\\Data1.xlsx", engine="openpyxl")

# Fill missing values
df = df.fillna(df.mean())

# Separate the features and the target variable
X = df.drop(columns=['Price of water (Rs./Kl)'])
y = df['Price of water (Rs./Kl)']

# Identify categorical features (you can change this list based on your actual dataset)
categorical_features = ['Type of treatment', 'Type of water']

# Verify the columns in X
print("Columns in DataFrame:", X.columns)

# Apply one-hot encoding to categorical features
encoder = OneHotEncoder(sparse_output=False, drop='first')
X_encoded = encoder.fit_transform(X[categorical_features])

# Convert the one-hot encoded features to a DataFrame
X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(categorical_features))

# Drop the original categorical columns from X
X = X.drop(columns=categorical_features)

# Concatenate the one-hot encoded features with the rest of the features
X = pd.concat([X.reset_index(drop=True), X_encoded_df.reset_index(drop=True)], axis=1)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize and train the Gradient Boosting model
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Get feature importances
feature_importances = model.feature_importances_

# Create a DataFrame for feature importances
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
})

# Sort the feature importances in descending order
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Print the predicted and actual prices
print("Predicted Prices:", y_pred[:5])
print("Actual Prices:", y_test[:5].values)
print("\nFeature Importances:")
print(importance_df)

# Save the model and the scaler
joblib.dump(model, "water_price_model_gb.pkl")
joblib.dump(scaler, "scaler.pkl")
