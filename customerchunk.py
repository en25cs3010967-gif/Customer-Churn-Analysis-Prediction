import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
file_name = r"C:\Users\tanis\.vscode\py.py\internship\WA_Fn-UseC_-Telco-Customer-Churn.csv"

df = pd.read_csv(file_name)

# Remove customer ID
df.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Remove missing values
df.dropna(inplace=True)

# Convert all categorical columns into numeric columns
df = pd.get_dummies(df, drop_first=True)

# Target column
target_column = "Churn_Yes"

X = df.drop(target_column, axis=1)
y = df[target_column]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Results
accuracy = accuracy_score(y_test, predictions)

print("\n========================")
print("MODEL PERFORMANCE")
print("========================")

print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Feature Importance
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Important Features:")
print(importance.head(10))
