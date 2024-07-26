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


model = tf.keras.Sequential([
    tf.keras.layers.Dense(264, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(232, activation='relu'),
    tf.keras.layers.Dense(1) 
])


model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])


history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, verbose=1)


test_loss, test_mae = model.evaluate(X_test, y_test)
print(f'Test MAE: {test_mae}')


model.save("water_price_model.h5")


joblib.dump(scaler, "scaler.pkl")