import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import sys
import traceback

print(f"--- Debug: Initial CWD: {os.getcwd()} ---")

# --- Define Paths ---
workspace_dir = os.getcwd()
mlruns_dir = os.path.join(workspace_dir, "mlruns")

# En Windows usamos la ruta directa sin file://
tracking_uri = mlruns_dir.replace("\\", "/")
if not tracking_uri.startswith("/"):
    tracking_uri = "/" + tracking_uri
tracking_uri = "file://" + tracking_uri

print(f"--- Debug: Tracking URI: {tracking_uri} ---")

os.makedirs(mlruns_dir, exist_ok=True)

# --- Configurar MLflow ---
mlflow.set_tracking_uri(tracking_uri)

# --- Crear o Establecer Experimento ---
experiment_name = "CI-CD-Lab2"
experiment_id = None
try:
    experiment_id = mlflow.create_experiment(name=experiment_name)
    print(f"--- Debug: Creado Experimento '{experiment_name}' con ID: {experiment_id} ---")
except mlflow.exceptions.MlflowException as e:
    if "RESOURCE_ALREADY_EXISTS" in str(e):
        print(f"--- Debug: Experimento '{experiment_name}' ya existe. ---")
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment:
            experiment_id = experiment.experiment_id
            print(f"--- Debug: ID del Experimento: {experiment_id} ---")
        else:
            print(f"--- ERROR: No se pudo obtener el experimento. ---")
            sys.exit(1)
    else:
        print(f"--- ERROR: {e} ---")
        raise e

if experiment_id is None:
    print(f"--- ERROR FATAL: No se pudo obtener un ID de experimento válido. ---")
    sys.exit(1)

# --- Cargar Datos y Entrenar Modelo ---
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)

# --- Guardar modelo localmente para validate.py ---
joblib.dump(model, "model.pkl")
print(f"--- Debug: Modelo guardado como model.pkl ---")

# --- Iniciar Run de MLflow ---
print(f"--- Debug: Iniciando run de MLflow en Experimento ID: {experiment_id} ---")
run = None
try:
    with mlflow.start_run(experiment_id=experiment_id) as run:
        run_id = run.info.run_id
        print(f"--- Debug: Run ID: {run_id} ---")

        mlflow.log_metric("mse", mse)
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("test_size", 0.2)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model"
        )
        print(f"✅ Modelo registrado correctamente. MSE: {mse:.4f}")

except Exception as e:
    print(f"\n--- ERROR durante la ejecución de MLflow ---")
    traceback.print_exc()
    sys.exit(1)