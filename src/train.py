import os
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# =====================
# CONFIG
# =====================
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

DATA_PATH = config["data_path"]
TARGET = config["target"]

# =====================
# MLflow (robusto en CI)
# =====================
# DB local + carpeta de artefactos local (sin rutas raras)
os.makedirs("artifacts", exist_ok=True)
mlflow.set_tracking_uri("sqlite:///mlflow.db")

EXPERIMENT_NAME = "wine-quality-exp"

exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if exp is None:
    mlflow.create_experiment(
        EXPERIMENT_NAME,
        artifact_location="file://" + os.path.abspath("artifacts"),
    )
mlflow.set_experiment(EXPERIMENT_NAME)

# =====================
# LOAD DATA
# =====================
df = pd.read_csv(DATA_PATH, sep=";")

# =====================
# PREPROCESS
# =====================
df[TARGET] = (df[TARGET] >= 6).astype(int)

X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =====================
# MODEL
# =====================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# =====================
# EVALUATION
# =====================
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# =====================
# LOGGING
# =====================
with mlflow.start_run():
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)

    mlflow.sklearn.log_model(model, name="model")

print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")
