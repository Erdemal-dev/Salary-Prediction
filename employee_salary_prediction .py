
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
data_path = "employee_salary_prediction_data.csv"
df = pd.read_csv(data_path)

# Splitting the data into features and target
X = df[["YearsExperience", "EducationLevel", "Age"]]
y = df["Salary"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and training the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions
predictions = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")

# Saving predictions to a CSV file
X_test["Actual Salary"] = y_test.values
X_test["Predicted Salary"] = predictions
X_test.to_csv("employee_salary_predictions.csv", index=False)
print("Predictions saved to employee_salary_predictions.csv")
