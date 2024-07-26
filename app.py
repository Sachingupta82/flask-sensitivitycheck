from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the model and scaler
model = joblib.load("water_price_model_gb.pkl")
scaler = joblib.load("scaler.pkl")

# Load the dataset to extract feature names
df = pd.read_excel("C:\\Users\\nicro\\OneDrive\\Desktop\\New folder\\Data1.xlsx", engine="openpyxl")
df = df.fillna(df.mean())
X = df.drop(columns=['Price of water (Rs./Kl)'])
feature_names = X.columns

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    request_data = request.json
    
    # Extract the 'data' field from the JSON payload
    data = request_data.get('data', None)
    
    # Validate the 'data' field
    if not isinstance(data, list) or len(data) != len(feature_names):
        return jsonify({"error": "Invalid input data"}), 400
    
    # Create a DataFrame with the input values and feature names
    features = pd.DataFrame([data], columns=feature_names)
    
    # Scale the input features
    features_scaled = scaler.transform(features)
    
    # Predict the price
    predicted_price = model.predict(features_scaled)[0]
    
    # Get feature importances
    feature_importances = model.feature_importances_
    
    # Create a DataFrame of feature importances
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)
    
    # Convert the DataFrame to a dictionary
    importance_dict = importance_df.to_dict(orient='records')
    
    # Prepare the response
    response = {
        'predicted_price': predicted_price,
        'feature_importances': importance_dict
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
