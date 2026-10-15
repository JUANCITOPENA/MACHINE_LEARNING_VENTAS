# Machine Learning aplicado a Ventas con Python

## Caso práctico

Este material contiene **5 casos de Machine Learning aplicados a un negocio de ventas**, utilizando datos ficticios para entrenamiento y demostración.

> **Importante:** los nombres de clientes y los registros son ficticios. Los nombres de personas/productos se utilizan únicamente como ejemplos realistas.

### Archivo de datos

El archivo `datasets_machine_learning_ventas.xlsx` contiene cinco hojas:

1. `01_Regresion_Ventas`
2. `02_Clientes_Clustering`
3. `03_Pedidos_SemiSupervisado`
4. `04_Reforzamiento`
5. `05_Autoencoder`

## Instalación

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### Linux/macOS

```bash
source .venv/bin/activate
```

Instalar librerías:

```bash
pip install pandas numpy matplotlib scikit-learn openpyxl tensorflow
```

---

# 1. Aprendizaje Supervisado — Regresión de ventas

## Objetivo

Predecir las ventas de una tienda utilizando:

- inversión en publicidad;
- porcentaje de descuento;
- tráfico de visitas.

## Algoritmo

**Regresión Lineal (`LinearRegression`)**

## Dataset

Hoja: `01_Regresion_Ventas`

Variable objetivo:

```text
Ventas_RDS
```

Variables predictoras:

```text
Publicidad_RDS
Descuento_Pct
Trafico_Visitas
```

## Código

```python
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Cargar datos
df = pd.read_excel(
    "datasets_machine_learning_ventas.xlsx",
    sheet_name="01_Regresion_Ventas"
)

X = df[[
    "Publicidad_RDS",
    "Descuento_Pct",
    "Trafico_Visitas"
]]

y = df["Ventas_RDS"]

# Separar entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Entrenar modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Predicción
predicciones = modelo.predict(X_test)

print("MAE:", round(mean_absolute_error(y_test, predicciones), 2))
print("R²:", round(r2_score(y_test, predicciones), 4))

# Predicción para una nueva situación
nuevo_dato = pd.DataFrame({
    "Publicidad_RDS": [15000],
    "Descuento_Pct": [10],
    "Trafico_Visitas": [300]
})

venta_estimada = modelo.predict(nuevo_dato)[0]

print(
    f"Ventas estimadas: RD$ {venta_estimada:,.2f}"
)

# Visualización
plt.scatter(y_test, predicciones)
plt.xlabel("Ventas reales")
plt.ylabel("Ventas predichas")
plt.title("Ventas reales vs. ventas predichas")
plt.show()
```

## Interpretación empresarial

La empresa puede utilizar este modelo para responder:

> "Si invierto RD$15,000 en publicidad, aplico 10% de descuento y espero 300 visitas, ¿cuánto podría vender?"

---

# 2. Aprendizaje No Supervisado — Segmentación de clientes

## Objetivo

Agrupar clientes según su comportamiento de compra sin utilizar una categoría previamente definida.

## Algoritmo

**K-Means Clustering**

## Variables

```text
Total_Compras_RDS
Frecuencia_Compras
Dias_Desde_Ultima_Compra
```

La columna `Segmento_Real_Referencia` existe solamente para comparar el resultado con los perfiles ficticios originales. **No debe utilizarse para entrenar el modelo.**

## Código

```python
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_excel(
    "datasets_machine_learning_ventas.xlsx",
    sheet_name="02_Clientes_Clustering"
)

features = [
    "Total_Compras_RDS",
    "Frecuencia_Compras",
    "Dias_Desde_Ultima_Compra"
]

X = df[features]

# Escalamiento
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means
modelo = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df["Cluster"] = modelo.fit_predict(X_scaled)

print(
    df[
        [
            "Cliente",
            "Total_Compras_RDS",
            "Frecuencia_Compras",
            "Dias_Desde_Ultima_Compra",
            "Cluster"
        ]
    ].head(20)
)

# Perfil promedio de cada cluster
perfil = df.groupby("Cluster")[features].mean()

print("\nPerfil de clusters:")
print(perfil)

# Visualización
plt.scatter(
    df["Total_Compras_RDS"],
    df["Frecuencia_Compras"],
    c=df["Cluster"]
)

plt.xlabel("Total de compras (RD$)")
plt.ylabel("Frecuencia de compras")
plt.title("Segmentación de clientes con K-Means")
plt.show()
```

## Interpretación empresarial

Los clusters pueden convertirse en segmentos como:

- **Premium**
- **Frecuentes**
- **Ocasionales**

La empresa podría entonces diseñar campañas diferentes para cada grupo.

---

# 3. Aprendizaje Semi-Supervisado — Clasificación de pedidos

## Objetivo

Clasificar pedidos como:

- `Alto Valor`
- `Normal`

utilizando pocos pedidos etiquetados y muchos pedidos sin etiqueta.

## Algoritmo

**Self-Training + Random Forest**

Este enfoque permite entrenar inicialmente con los datos etiquetados y generar pseudo-etiquetas para algunos registros con alta confianza.

## Dataset

Hoja:

```text
03_Pedidos_SemiSupervisado
```

La columna `Etiqueta` contiene aproximadamente 30% de registros etiquetados.

## Código

```python
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.semi_supervised import SelfTrainingClassifier

df = pd.read_excel(
    "datasets_machine_learning_ventas.xlsx",
    sheet_name="03_Pedidos_SemiSupervisado"
)

features = [
    "Monto_RDS",
    "Cantidad_Items"
]

X = df[features]

# Convertir etiquetas:
# Alto Valor = 1
# Normal = 0
# Sin etiqueta = -1

y = df["Etiqueta"].map({
    "Alto Valor": 1,
    "Normal": 0
}).fillna(-1)

base_model = RandomForestClassifier(
    n_estimators=150,
    random_state=42,
    class_weight="balanced"
)

modelo = SelfTrainingClassifier(
    estimator=base_model,
    threshold=0.80
)

modelo.fit(X, y)

df["Prediccion"] = modelo.predict(X)

df["Prediccion_Texto"] = df["Prediccion"].map({
    1: "Alto Valor",
    0: "Normal"
})

print(
    df[
        [
            "Pedido_ID",
            "Monto_RDS",
            "Cantidad_Items",
            "Etiqueta",
            "Prediccion_Texto"
        ]
    ].head(30)
)
```

## Interpretación empresarial

Puede utilizarse para detectar pedidos importantes cuando la empresa solamente ha podido revisar manualmente una pequeña parte de sus transacciones.

Ejemplo:

```text
Pedido:
Monto = RD$120,000
Items = 8

Predicción:
ALTO VALOR
```

Esto puede ayudar a:

- priorizar atención;
- asignar vendedores;
- controlar entregas;
- ofrecer beneficios a clientes de alto valor.

---

# 4. Aprendizaje por Refuerzo — Optimización de promociones

## Objetivo

Crear un agente que aprenda qué nivel de descuento conviene aplicar según el contexto de venta.

## Algoritmo

**Q-Learning**

El agente observa un estado y elige una acción.

### Estado

```text
Día
Producto
Temporada
```

### Acciones

```text
0% descuento
10% descuento
20% descuento
30% descuento
```

### Recompensa

Una recompensa sencilla puede ser el ingreso generado:

```text
recompensa = ingreso
```

Para una implementación real habría que considerar también margen, inventario, costo y otras restricciones.

## Código

```python
import numpy as np
import pandas as pd

df = pd.read_excel(
    "datasets_machine_learning_ventas.xlsx",
    sheet_name="04_Reforzamiento"
)

# Acciones posibles
acciones = [0, 10, 20, 30]

# Estados simplificados
estados = [
    f"{dia}_{producto}_{temporada}"
    for dia in df["Dia"].unique()
    for producto in df["Producto"].unique()
    for temporada in df["Temporada"].unique()
]

estado_id = {
    estado: i
    for i, estado in enumerate(estados)
}

Q = np.zeros(
    (len(estados), len(acciones))
)

alpha = 0.10
gamma = 0.90
epsilon = 0.20

for episodio in range(5000):

    fila = df.sample(1).iloc[0]

    estado = (
        f"{fila['Dia']}_"
        f"{fila['Producto']}_"
        f"{fila['Temporada']}"
    )

    s = estado_id[estado]

    # Exploración / explotación
    if np.random.random() < epsilon:
        a = np.random.randint(len(acciones))
    else:
        a = np.argmax(Q[s])

    descuento = acciones[a]

    # Simulación sencilla de demanda
    demanda = fila["Demanda_Base"]

    ventas_unidades = max(
        0,
        demanda * (
            1 + descuento / 100
        ) * np.random.uniform(0.75, 1.15)
    )

    ingreso = (
        ventas_unidades
        * fila["Precio_RDS"]
        * (1 - descuento / 100)
    )

    recompensa = ingreso

    # Actualización Q-Learning
    siguiente_estado = s

    Q[s, a] = Q[s, a] + alpha * (
        recompensa
        + gamma * np.max(Q[siguiente_estado])
        - Q[s, a]
    )

# Mostrar política aprendida
print("Política aprendida:")

for i, estado in enumerate(estados[:20]):
    mejor_accion = acciones[np.argmax(Q[i])]

    print(
        f"{estado} -> "
        f"Descuento recomendado: "
        f"{mejor_accion}%"
    )
```

## Interpretación empresarial

El concepto puede evolucionar hacia un sistema que decida:

> "Para este producto, este día y esta temporada, ¿qué promoción maximiza el resultado?"

En producción, la recompensa debería ser **margen o utilidad**, no únicamente ingresos.

---

# 5. Aprendizaje Auto-Supervisado — Detección de patrones/anomalías

## Objetivo

Aprender una representación de clientes utilizando datos sin etiquetas.

Una aplicación práctica es identificar comportamientos atípicos que posteriormente pueden ser investigados.

## Algoritmo

**Autoencoder**

Un autoencoder aprende a reconstruir sus propios datos.

Cuando un registro es muy diferente del comportamiento aprendido, normalmente presenta un mayor error de reconstrucción.

## Código

```python
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense

df = pd.read_excel(
    "datasets_machine_learning_ventas.xlsx",
    sheet_name="05_Autoencoder"
)

features = [
    "Compras_90d",
    "Monto_90d_RDS",
    "Dias_Activo",
    "Productos_Distintos",
    "Devoluciones",
    "Visitas_Web"
]

X = df[features].copy()

# Escalar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Autoencoder
input_dim = X_scaled.shape[1]

entrada = Input(shape=(input_dim,))

encoded = Dense(
    8,
    activation="relu"
)(entrada)

encoded = Dense(
    3,
    activation="relu"
)(encoded)

decoded = Dense(
    8,
    activation="relu"
)(encoded)

decoded = Dense(
    input_dim,
    activation="linear"
)(decoded)

autoencoder = Model(
    entrada,
    decoded
)

autoencoder.compile(
    optimizer="adam",
    loss="mse"
)

autoencoder.fit(
    X_scaled,
    X_scaled,
    epochs=80,
    batch_size=32,
    validation_split=0.20,
    verbose=1
)

# Reconstrucción
reconstruido = autoencoder.predict(
    X_scaled,
    verbose=0
)

# Error de reconstrucción
error = np.mean(
    np.square(
        X_scaled - reconstruido
    ),
    axis=1
)

df["Error_Reconstruccion"] = error

# Umbral
umbral = np.quantile(
    error,
    0.95
)

df["Anomalia"] = (
    df["Error_Reconstruccion"] > umbral
)

print(
    df.sort_values(
        "Error_Reconstruccion",
        ascending=False
    ).head(20)
)
```

## Interpretación empresarial

Las observaciones con errores elevados pueden investigarse como posibles:

- comportamientos de compra inusuales;
- clientes atípicos;
- errores de captura;
- operaciones que requieren revisión;
- patrones potencialmente relacionados con fraude.

**Un modelo de anomalías no demuestra fraude por sí mismo.** Solo identifica comportamientos que merecen investigación.

---

# Resumen

| Tipo | Algoritmo | Problema de ventas |
|---|---|---|
| Supervisado | Regresión Lineal | Predecir ventas |
| No supervisado | K-Means | Segmentar clientes |
| Semi-supervisado | Self-Training + Random Forest | Clasificar pedidos con pocas etiquetas |
| Por refuerzo | Q-Learning | Optimizar promociones |
| Auto-supervisado | Autoencoder | Detectar patrones/anomalías |

## Flujo recomendado

```text
Excel / CSV
    ↓
Pandas
    ↓
Limpieza y preparación
    ↓
EDA
    ↓
Feature Engineering
    ↓
Machine Learning
    ↓
Evaluación
    ↓
Predicciones
    ↓
Dashboard / Power BI / Streamlit
    ↓
Decisión empresarial
```

## Librerías principales

```text
pandas
numpy
scikit-learn
matplotlib
openpyxl
tensorflow
```

## Próximo nivel

Estos ejemplos pueden ampliarse con:

- validación cruzada;
- GridSearchCV;
- XGBoost;
- Random Forest;
- Gradient Boosting;
- Prophet para series temporales;
- redes neuronales;
- SHAP para explicar modelos;
- Streamlit para crear una aplicación;
- Power BI para visualizar resultados;
- SQL Server como fuente de datos.
