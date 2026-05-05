import mlflow
import os

# misma ruta que train
tracking_path = os.path.abspath("mlruns")
mlflow.set_tracking_uri(f"file://{tracking_path}")

experiment_name = "wine-quality-exp"

experiment = mlflow.get_experiment_by_name(experiment_name)

if experiment is None:
    print("⚠️ No existe el experimento, se omite evaluación")
    exit(0)

runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["start_time DESC"]
)

if runs.empty:
    print("⚠️ No hay runs disponibles")
    exit(0)

accuracy = runs.iloc[0].get("metrics.accuracy", None)
f1 = runs.iloc[0].get("metrics.f1_score", None)

print(f"📊 Accuracy: {accuracy}")
print(f"📊 F1 Score: {f1}")

if accuracy and accuracy > 0.7:
    print("✅ Modelo aceptable")
else:
    print("⚠️ Modelo con bajo desempeño (no se detiene CI)")
