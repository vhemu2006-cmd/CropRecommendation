import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Read dataset
df = pd.read_csv("dataset/Crop_recommendation.csv")

# Features
X = df.drop("label", axis=1)

# Target
y = df["label"]

# Preprocessing
scaler = StandardScaler()

X = scaler.fit_transform(X)

# Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model

model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)

# Prediction

pred = model.predict(X_test)

print("Accuracy :", accuracy_score(y_test, pred))

# Save model

joblib.dump(model, "model/crop_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")