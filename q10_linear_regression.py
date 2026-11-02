import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("Iris.csv")

X = df[["SepalLengthCm", "SepalWidthCm", "PetalWidthCm"]]
y = df["PetalLengthCm"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

print("Model Coefficients:")
for name, coef in zip(X.columns, model.coef_):
    print(f"  {name}: {coef:.4f}")
print(f"  Intercept: {model.intercept_:.4f}")

y_pred = model.predict(X_test)

results = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred.round(2)})
print("\nPredictions on test data:")
print(results.to_string(index=False))
