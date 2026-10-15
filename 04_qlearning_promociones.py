# 04 - Aprendizaje por Refuerzo: Q-Learning
# Optimización de promociones
# Ejecutar en VS Code:
# python 04_qlearning_promociones.py

import numpy as np
import pandas as pd

ARCHIVO = "datasets_machine_learning_ventas.xlsx"
HOJA = "04_Reforzamiento"

df = pd.read_excel(ARCHIVO, sheet_name=HOJA)

# Acciones posibles del agente.
acciones = [0, 10, 20, 30]

# Crear estados combinando contexto comercial.
estados = [
    f"{dia}|{producto}|{temporada}"
    for dia in df["Dia"].unique()
    for producto in df["Producto"].unique()
    for temporada in df["Temporada"].unique()
]

estado_id = {estado: i for i, estado in enumerate(estados)}

Q = np.zeros((len(estados), len(acciones)))

alpha = 0.10       # tasa de aprendizaje
gamma = 0.90       # importancia de recompensas futuras
epsilon = 0.20     # exploración

rng = np.random.default_rng(42)

print("=" * 60)
print("Q-LEARNING - OPTIMIZACIÓN DE PROMOCIONES")
print("=" * 60)
print(f"Estados: {len(estados)}")
print(f"Acciones: {acciones}")

for episodio in range(5000):

    fila = df.sample(1, random_state=episodio).iloc[0]

    estado = (
        f"{fila['Dia']}|"
        f"{fila['Producto']}|"
        f"{fila['Temporada']}"
    )

    s = estado_id[estado]

    # Exploración vs explotación
    if rng.random() < epsilon:
        a = rng.integers(len(acciones))
    else:
        a = np.argmax(Q[s])

    descuento = acciones[a]

    # Simulación de comportamiento del cliente.
    demanda = fila["Demanda_Base"]

    ventas_unidades = max(
        0,
        demanda
        * (1 + descuento / 100)
        * rng.uniform(0.75, 1.15)
    )

    ingreso = (
        ventas_unidades
        * fila["Precio_RDS"]
        * (1 - descuento / 100)
    )

    recompensa = ingreso

    siguiente_estado = s

    Q[s, a] = Q[s, a] + alpha * (
        recompensa
        + gamma * np.max(Q[siguiente_estado])
        - Q[s, a]
    )

print("\nPOLÍTICA APRENDIDA")
print("-" * 60)

resultados = []

for estado, s in estado_id.items():

    mejor_accion = acciones[np.argmax(Q[s])]

    dia, producto, temporada = estado.split("|")

    resultados.append({
        "Dia": dia,
        "Producto": producto,
        "Temporada": temporada,
        "Descuento_Recomendado_Pct": mejor_accion,
        "Valor_Q": Q[s, np.argmax(Q[s])]
    })

resultado = pd.DataFrame(resultados)

print(resultado.head(30).to_string(index=False))

resultado.to_excel(
    "politica_promociones_qlearning.xlsx",
    index=False
)

print("\nArchivo generado: politica_promociones_qlearning.xlsx")

print("\nNOTA:")
print("Este ejemplo utiliza una simulación educativa.")
print("En producción, la recompensa debería considerar margen/utilidad,")
print("inventario, costo, elasticidad de demanda y restricciones comerciales.")
