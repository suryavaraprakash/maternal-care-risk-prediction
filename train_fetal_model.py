import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load Dataset
df = pd.read_csv("fetal_health.csv")

# Features
X = df.drop("fetal_health", axis=1)

# Target
y = df["fetal_health"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train Model
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)

print(f"Accuracy: {accuracy:.4f}")

# Save Model
with open("model/fetal_health_classifier.sav", "wb") as f:
    pickle.dump(model, f)

print("New fetal_health_classifier.sav saved successfully!")