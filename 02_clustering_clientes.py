# 02 - Aprendizaje No Supervisado: K-Means
# Ejecutar en VS Code:
# python 02_clustering_clientes.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

ARCHIVO = "datasets_machine_learning_ventas.xlsx"
HOJA = "02_Clientes_Clustering"

df = pd.read_excel(ARCHIVO, sheet_name=HOJA)

features = [
    "Total_Compras_RDS",
    "Frecuencia_Compras",
    "Dias_Desde_Ultima_Compra"
]

X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

modelo = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df["Cluster"] = modelo.fit_predict(X_scaled)

silhouette = silhouette_score(X_scaled, df["Cluster"])

print("=" * 60)
print("K-MEANS - SEGMENTACIÓN DE CLIENTES")
print("=" * 60)
print(f"Clientes: {len(df)}")
print(f"Silhouette Score: {silhouette:.4f}")

print("\nCANTIDAD DE CLIENTES POR CLUSTER")
print(df["Cluster"].value_counts().sort_index())

perfil = df.groupby("Cluster")[features].mean().round(2)

print("\nPERFIL PROMEDIO")
print(perfil)

# Dar nombres interpretables a los clusters según su comportamiento
ranking = perfil["Total_Compras_RDS"].sort_values(ascending=False)
nombres_cluster = {}

for posicion, cluster in enumerate(ranking.index):
    if posicion == 0:
        nombres_cluster[cluster] = "Premium"
    elif posicion == 1:
        nombres_cluster[cluster] = "Frecuente"
    else:
        nombres_cluster[cluster] = "Ocasional"

df["Segmento_Modelo"] = df["Cluster"].map(nombres_cluster)

print("\nEJEMPLO DE CLIENTES SEGMENTADOS")
print(
    df[
        [
            "Cliente_ID",
            "Cliente",
            "Total_Compras_RDS",
            "Frecuencia_Compras",
            "Dias_Desde_Ultima_Compra",
            "Segmento_Modelo"
        ]
    ].head(20).to_string(index=False)
)

df.to_excel("clientes_segmentados_resultado.xlsx", index=False)
print("\nArchivo generado: clientes_segmentados_resultado.xlsx")

plt.figure(figsize=(10, 6))
plt.scatter(
    df["Total_Compras_RDS"],
    df["Frecuencia_Compras"],
    c=df["Cluster"],
    alpha=0.75
)
plt.xlabel("Total de compras (RD$)")
plt.ylabel("Frecuencia de compras")
plt.title("Segmentación de clientes con K-Means")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()
