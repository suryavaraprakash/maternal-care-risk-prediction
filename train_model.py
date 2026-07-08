import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
print("Training script started...")
# Load Dataset
df = pd.read_csv("Maternal Health Risk Data Set.csv")

# Features (ALL 6 FEATURES)
X = df[[
    "Age",
    "SystolicBP",
    "DiastolicBP",
    "BS",
    "BodyTemp",
    "HeartRate"
]]

# Target
y = df["RiskLevel"]

# Convert labels into numbers
y = y.map({
    "low risk": 0,
    "mid risk": 1,
    "high risk": 2
})

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Random Forest
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)

print("Accuracy:", accuracy)

# Save scaler
pickle.dump(
    scaler,
    open("model/scaler_maternal_model.sav", "wb")
)

# Save model
pickle.dump(
    model,
    open("model/finalized_maternal_model.sav", "wb")
)

print("Training Completed Successfully!")
print("New Model Saved.")