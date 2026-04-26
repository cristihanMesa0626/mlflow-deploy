import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
import sys
import os

THRESHOLD = 5000.0

print("--- Debug: Cargando dataset load_diabetes ---")
X, y = load_diabetes(return_X_y=True, as_frame=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"--- Debug: Dimensiones de X_test: {X_test.shape} ---")

model_filename = "model.pkl"
model_path = os.path.abspath(os.path.join(os.getcwd(), model_filename))
print(f"--- Debug: Intentando cargar modelo desde: {model_path} ---")

try:
    model = joblib.load(model_path)
except FileNotFoundError:
    print(f"--- ERROR: No se encontró el modelo en '{model_path}'. ---")
    print(f"--- Debug: Archivos en {os.getcwd()}: ---")
    print(os.listdir(os.getcwd()))
    sys.exit(1)

print("--- Debug: Realizando predicciones ---")
try:
    y_pred = model.predict(X_test)
except ValueError as pred_err:
    print(f"--- ERROR durante la predicción: {pred_err} ---")
    sys.exit(1)

mse = mean_squared_error(y_test, y_pred)
print(f"🔍 MSE del modelo: {mse:.4f} (umbral: {THRESHOLD})")

if mse <= THRESHOLD:
    print("✅ El modelo cumple los criterios de calidad.")
    sys.exit(0)
else:
    print("❌ El modelo no cumple el umbral. Deteniendo pipeline.")
    sys.exit(1)