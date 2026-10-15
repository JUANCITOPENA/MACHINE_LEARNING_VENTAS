# 01 - Aprendizaje Supervisado: Regresión de Ventas
# Ejecutar en VS Code:
# python 01_regresion_ventas.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

ARCHIVO = "datasets_machine_learning_ventas.xlsx"
HOJA = "01_Regresion_Ventas"

df = pd.read_excel(ARCHIVO, sheet_name=HOJA)

X = df[["Publicidad_RDS", "Descuento_Pct", "Trafico_Visitas"]]
y = df["Ventas_RDS"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

modelo = LinearRegression()
modelo.fit(X_train, y_train)

predicciones = modelo.predict(X_test)

mae = mean_absolute_error(y_test, predicciones)
r2 = r2_score(y_test, predicciones)

print("=" * 60)
print("REGRESIÓN LINEAL - PREDICCIÓN DE VENTAS")
print("=" * 60)
print(f"Registros: {len(df)}")
print(f"MAE: RD$ {mae:,.2f}")
print(f"R²: {r2:.4f}")

nuevo_dato = pd.DataFrame({
    "Publicidad_RDS": [15000],
    "Descuento_Pct": [10],
    "Trafico_Visitas": [300]
})

venta_estimada = modelo.predict(nuevo_dato)[0]

print("\nNUEVA PREDICCIÓN")
print(f"Publicidad: RD$ 15,000")
print("Descuento: 10%")
print("Tráfico: 300 visitas")
print(f"Ventas estimadas: RD$ {venta_estimada:,.2f}")

print("\nCOEFICIENTES")
for variable, coef in zip(X.columns, modelo.coef_):
    print(f"{variable}: {coef:,.4f}")

plt.figure(figsize=(9, 6))
plt.scatter(y_test, predicciones, alpha=0.75)
plt.xlabel("Ventas reales (RD$)")
plt.ylabel("Ventas predichas (RD$)")
plt.title("Regresión Lineal: Ventas reales vs. predichas")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()
