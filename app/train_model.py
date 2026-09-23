import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("dataset/UNSW_NB15_training-set.csv")

# Remove unnecessary columns
df = df.drop(columns=["id", "attack_cat"])

# Separate features and target
X = df.drop(columns=["label"])
y = df["label"]

# Identify categorical and numerical columns
categorical_columns = X.select_dtypes(include=["object"]).columns
numerical_columns = X.select_dtypes(exclude=["object"]).columns

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical_columns
        ),
        (
            "numerical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median"))
            ]),
            numerical_columns
        )
    ]
)

# Machine learning model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# Complete pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training model...")

# Train
pipeline.fit(X_train, y_train)

# Predict
predictions = pipeline.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Save model
joblib.dump(pipeline, "models/intrusion_detection_model.pkl")

print("\nModel saved successfully!")