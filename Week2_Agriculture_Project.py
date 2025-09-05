# Week 2 - Sustainable Agriculture Project
# Theme: Model Selection & Building for Crop Yield Prediction

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
# Step 1: Load Dataset
print("Loading the dataset...")
data = pd.read_csv("sustainable_agriculture.csv")
print("\nDataset preview:")
print(data.head())
# Step 2: Feature Selection
# Using Rainfall and Fertilizer as features to predict Crop Yield
X = data[["Rainfall_mm", "Fertilizer_kg_per_hectare"]]
y = data["Crop_Yield_tons_per_hectare"]
# Step 3: Split Data into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("\nTraining and Testing split done.")
# Step 4: Model Building
model = LinearRegression()
model.fit(X_train, y_train)
print("\nModel trained successfully.")
# Step 5: Predictions
y_pred = model.predict(X_test)
# Step 6: Evaluate Model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\nModel Evaluation:")
print("Mean Squared Error:", mse)
print("R2 Score:", r2)
# Step 7: Plot Actual vs Predicted
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Crop Yield")
plt.ylabel("Predicted Crop Yield")
plt.title("Actual vs Predicted Crop Yield")
plt.show()
# Step 8: Model Coefficients
print("\nModel Coefficients:")
print("Rainfall Coefficient:", model.coef_[0])
print("Fertilizer Coefficient:", model.coef_[1])
print("Intercept:", model.intercept_)
