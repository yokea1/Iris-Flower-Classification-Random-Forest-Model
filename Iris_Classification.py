import pandas as pd
df = pd.read_csv("Iris.csv")  # no need to add path if in the same directory
print(df.head())

# === Rename columns to standard English variable names ===
df = df.rename(columns={
    "SepalLengthCm": "sepal_length",
    "SepalWidthCm": "sepal_width",
    "PetalLengthCm": "petal_length",
    "PetalWidthCm": "petal_width",
    "Species": "species"
})

# === View the processed data ===
print(df.head())                 # first few rows
print(df.info())                 # data types + missing value check
print(df.isnull().sum())         # missing value count per column
print(df["species"].value_counts())  # count per species

from sklearn.preprocessing import LabelEncoder, StandardScaler

# Convert species to numeric labels (e.g., 0, 1, 2)
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["species"])

# Print label mapping
print("Label mapping:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

# Extract features and labels
X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
y = df["label"]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# View standardized data
print(pd.DataFrame(X_scaled, columns=X.columns).head())

from sklearn.model_selection import train_test_split

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print("Number of training samples:", len(X_train))
print("Number of test samples:", len(X_test))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Create and train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate the model
print("🎯 Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))

import matplotlib.pyplot as plt

importances = model.feature_importances_
features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

plt.bar(features, importances)
plt.title("Feature Importances")
plt.ylabel("Importance Score")
plt.show()