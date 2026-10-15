# 05 - Aprendizaje Auto-Supervisado: Autoencoder
# Detección de comportamientos atípicos
# Ejecutar en VS Code:
# python 05_autoencoder_anomalias.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

ARCHIVO = "datasets_machine_learning_ventas.xlsx"
HOJA = "05_Autoencoder"

df = pd.read_excel(ARCHIVO, sheet_name=HOJA)

features = [
    "Compras_90d",
    "Monto_90d_RDS",
    "Dias_Activo",
    "Productos_Distintos",
    "Devoluciones",
    "Visitas_Web"
]

X = df[features].copy()

print("=" * 60)
print("AUTOENCODER - DETECCIÓN DE ANOMALÍAS")
print("=" * 60)
print(f"Registros: {len(df)}")
print(f"Variables: {len(features)}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nEntrenando autoencoder...")

autoencoder = MLPRegressor(
    hidden_layer_sizes=(8, 3, 8),
    activation="relu",
    solver="adam",
    max_iter=80,
    batch_size=32,
    random_state=42,
    verbose=True
)

autoencoder.fit(X_scaled, X_scaled)

reconstruido = autoencoder.predict(X_scaled)

error = np.mean(
    np.square(X_scaled - reconstruido),
    axis=1
)

df["Error_Reconstruccion"] = error

# Marcamos aproximadamente el 5% superior como candidatos a anomalía.
umbral = np.quantile(error, 0.95)

df["Anomalia"] = (
    df["Error_Reconstruccion"] > umbral
)

print(f"\nUmbral de anomalía: {umbral:.4f}")
print(
    f"Anomalías detectadas: "
    f"{df['Anomalia'].sum()}"
)

resultado = df.sort_values(
    "Error_Reconstruccion",
    ascending=False
)

print("\nTOP 20 COMPORTAMIENTOS MÁS ATÍPICOS")
print(
    resultado[
        [
            "Cliente_ID",
            "Compras_90d",
            "Monto_90d_RDS",
            "Dias_Activo",
            "Productos_Distintos",
            "Devoluciones",
            "Visitas_Web",
            "Error_Reconstruccion",
            "Anomalia"
        ]
    ].head(20).to_string(index=False)
)

resultado.to_excel(
    "anomalias_clientes_autoencoder.xlsx",
    index=False
)

print("\nArchivo generado: anomalias_clientes_autoencoder.xlsx")
print("\nIMPORTANTE:")
print("Una anomalía no significa automáticamente fraude.")
print("Representa un comportamiento que merece investigación.")
