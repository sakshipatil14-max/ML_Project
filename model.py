import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load data
df = pd.read_csv("data.csv")

print(df)

# Split data
X = df[['Hours']]
y = df['Marks']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

print("Predictions:", y_pred)

# Graph
plt.scatter(X, y)
plt.plot(X_test, y_pred)
plt.xlabel("Hours")
plt.ylabel("Marks")
plt.title("Prediction Model")
plt.show()