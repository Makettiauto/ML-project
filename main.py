import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

#DATA
data = pd.read_csv('synthetic_data.csv')
data = data.dropna()

features = ["study_hours_per_day", "screen_hours", "exercise_frequency", "mental_health_rating", "sleep_hours", "screen_hours"]

data = data.dropna(subset=["study_hours_per_day", "social_media_hours", "netflix_hours"
                           , "exercise_frequency", "mental_health_rating", "sleep_hours", "exam_score"])
data["screen_hours"] = data["social_media_hours"] + data["netflix_hours"]

#Features X and target y
X = data[features]
y = data["exam_score"]


#Split data into training and testing/validation sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)

#Split testing/validation set into testing and validation sets
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

#Train Linear Regression model
print("Training Linear Regression model...")
model = LinearRegression()
model.fit(X_train, y_train)

#Validate model
val_preds = model.predict(X_val)
val_mse = mean_squared_error(y_val, val_preds)
val_r2 = r2_score(y_val, val_preds)
print("Validation -----------------------------")
print("Validation MSE:", val_mse)
print("Validation R²:", val_r2)

#Make predictions
predictions = model.predict(X)

#Model parameters
print("Results ------------------------------")
print("Model trained. Here are the coefficients (value > 0 means positive correlation, value < 0 means negative correlation):")
print("Coefficients:", model.coef_)

print("Intercept:", model.intercept_)

#Evaluate model on test set
test_preds = model.predict(X_test)
test_mse = mean_squared_error(y_test, test_preds)
test_r2 = r2_score(y_test, test_preds)
print("Test results-----------------------------")
print("Test MSE:", test_mse)
print("Test R²:", test_r2)

#Correaltion matrix
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()