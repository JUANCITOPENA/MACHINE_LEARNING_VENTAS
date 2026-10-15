# 03 - Aprendizaje Semi-Supervisado
# Self-Training + Random Forest
# Ejecutar en VS Code:
# python 03_semi_supervisado_pedidos.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.semi_supervised import SelfTrainingClassifier

ARCHIVO = "datasets_machine_learning_ventas.xlsx"
HOJA = "03_Pedidos_SemiSupervisado"

df = pd.read_excel(ARCHIVO, sheet_name=HOJA)

features = ["Monto_RDS", "Cantidad_Items"]
X = df[features]

# Las etiquetas conocidas se convierten a 0/1.
# Los pedidos sin etiqueta se representan como -1.
y = df["Etiqueta"].map({
    "Alto Valor": 1,
    "Normal": 0
}).fillna(-1).astype(int)

print("=" * 60)
print("APRENDIZAJE SEMI-SUPERVISADO")
print("=" * 60)

print(f"Total de pedidos: {len(df)}")
print(f"Pedidos etiquetados: {(y != -1).sum()}")
print(f"Pedidos sin etiqueta: {(y == -1).sum()}")

modelo_base = RandomForestClassifier(
    n_estimators=150,
    random_state=42,
    class_weight="balanced"
)

modelo = SelfTrainingClassifier(
    estimator=modelo_base,
    threshold=0.80
)

modelo.fit(X, y)

df["Prediccion"] = modelo.predict(X)

df["Prediccion_Texto"] = df["Prediccion"].map({
    1: "Alto Valor",
    0: "Normal"
})

# El modelo puede proporcionar probabilidades para los registros.
probabilidades = modelo.predict_proba(X)
df["Confianza"] = probabilidades.max(axis=1)

print("\nRESULTADOS")
print(
    df[
        [
            "Pedido_ID",
            "Monto_RDS",
            "Cantidad_Items",
            "Etiqueta",
            "Prediccion_Texto",
            "Confianza"
        ]
    ].head(30).to_string(index=False)
)

df.to_excel("pedidos_clasificados_resultado.xlsx", index=False)
print("\nArchivo generado: pedidos_clasificados_resultado.xlsx")
