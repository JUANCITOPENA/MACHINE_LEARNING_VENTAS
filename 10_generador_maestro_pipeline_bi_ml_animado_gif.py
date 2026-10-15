# ==============================================================================
# 10 - GENERADOR MAESTRO UNIFICADO DE PIPELINE BI & MACHINE LEARNING (.GIF) 16:9
# Suite Completa de 20 Gráficos Animados en 3 Hojas Ejecutivas
# Hoja 1: Tablero de KPIs Comerciales (8 Gráficos, Matriz 4x2)
# Hoja 2: Estadísticos & Machine Learning (6 Gráficos, Matriz 3x2)
# Hoja 3: Metas & Cumplimiento (6 Gráficos, Matriz 3x2)
# + Suite Maestra Combinada Todo-en-Uno (20 Gráficos en Transición Continua)
# Stack: Pandas, NumPy, Matplotlib & Scikit-Learn
# Autor: Ing. Juancito Peña & Antigravity
# Ejecutar: python 10_generador_maestro_pipeline_bi_ml_animado_gif.py
# ==============================================================================

import os
import io
import sys
import time
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Renderizado de alta velocidad sin dependencias de GUI
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Machine Learning Modules
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

warnings.filterwarnings("ignore")

# Estilización tipográfica y estética ejecutiva
plt.rcParams['font.sans-serif'] = 'Segoe UI'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#cbd5e1'
plt.rcParams['axes.linewidth'] = 0.9

# Nombres de archivos de salida
H1_GIF = "dashboard_10_hoja1_kpis_comerciales_16_9.gif"
H1_PNG = "dashboard_10_hoja1_kpis_comerciales_16_9.png"
H2_GIF = "dashboard_10_hoja2_modelos_ml_16_9.gif"
H2_PNG = "dashboard_10_hoja2_modelos_ml_16_9.png"
H3_GIF = "dashboard_10_hoja3_metas_cumplimiento_16_9.gif"
H3_PNG = "dashboard_10_hoja3_metas_cumplimiento_16_9.png"
MASTER_GIF = "dashboard_10_master_pipeline_completo_16_9.gif"
EXCEL_OUTPUT = "pipeline_transaccional_bi_ml_1000.xlsx"

NUM_FRAMES = 22  # Fotogramas clave para fluidez óptima y tamaño de archivo ligero

print("=" * 80)
print("[+] SCRIPT 10: PIPELINE INDUSTRIAL DE BI & MACHINE LEARNING ANIMADO (.GIF)")
print("    20 Graficos Ejecutivos en 3 Dashboards 16:9 + Suite Maestra Completa")
print("=" * 80)

t_inicio = time.time()

# ==============================================================================
# FASE 1: GENERACIÓN DEL DATASET & FEATURE ENGINEERING EN MEMORIA
# ==============================================================================
print("\n[1/5] Generando dataset deterministico de 1,000 transacciones y modelos ML...")

random.seed(42)
np.random.seed(42)

productos = ["Hamburguesa", "Pizza", "Refresco", "Papas Fritas", "Pollo Frito",
             "Café", "Empanada", "Hot Dog", "Helado", "Sandwich"]

categorias = [
    "Comida Rápida",
    "Bebidas",
    "Postres",
    "Snacks",
    "Café y Té"
]

clientes = ["Juan", "Ana", "Pedro", "Luis", "Maria", "Jose", "Carla",
            "Miguel", "Laura", "Andres", "Sofia", "Daniel", "Paula", "Ricardo", "Elena"]

vendedores = ["Carlos", "Beatriz", "Manuel", "Patricia", "Roberto"]
condiciones = ["Contado", "Credito", "Transferencia", "Cheque"]
regiones = ["Este", "Norte", "Sur", "Cibao"]

fechas_disponibles = pd.date_range("2025-01-01", "2026-09-17", freq="D")
n = 1000

df = pd.DataFrame({
    "Fecha": np.random.choice(fechas_disponibles, size=n),
    "Producto": random.choices(productos, k=n),
    "Categoria": random.choices(categorias, k=n),
    "Cliente": random.choices(clientes, k=n),
    "Vendedor": random.choices(vendedores, k=n),
})

df["Mes"] = df["Fecha"].dt.month
df["Año"] = df["Fecha"].dt.year
df["Precio_Compra"] = np.random.randint(20, 200, size=n).astype(float)
df["Precio_Venta"] = df["Precio_Compra"] + np.random.randint(20, 300, size=n)
df["Cantidad"] = np.random.randint(1, 10, size=n)

meta_vendedor = {v: random.randint(15000, 30000) for v in vendedores}
df["Meta_Vendedor"] = df["Vendedor"].map(meta_vendedor)

cumplimiento_min = 0.85
cumplimiento_max = 1.20
df["Monto"] = [
    round(meta_vendedor[v] * random.uniform(cumplimiento_min, cumplimiento_max), 2)
    for v in df["Vendedor"]
]

df["Costo"] = (df["Precio_Compra"] * df["Cantidad"]).round(2)
df["Margen"] = (df["Monto"] - df["Costo"]).round(2)
df["%Margen"] = ((df["Margen"] / df["Monto"]) * 100).round(2)

df["CondicionPago"] = np.random.choice(condiciones, size=n, p=[0.45, 0.25, 0.18, 0.12])
df["Region"] = np.random.choice(regiones, size=n, p=[0.48, 0.22, 0.18, 0.12])

df["%Participacion"] = ((df["Monto"] / df["Monto"].sum()) * 100).round(2)
df["%CumplimientoMeta"] = ((df["Monto"] / df["Meta_Vendedor"]) * 100).round(2)

# Exportar dataset para respaldo y auditoría
df.to_excel(EXCEL_OUTPUT, index=False)
print(f"   [OK] Dataset maestro persistido en: {EXCEL_OUTPUT}")

# --- Cálculos y Modelos para Hoja 1 (KPIs Comerciales) ---
ventas_ym = df.groupby(["Año", "Mes"])["Monto"].sum().reset_index()
pivot_ym = ventas_ym.pivot(index="Mes", columns="Año", values="Monto").fillna(0)
meses_abrev = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]

cond_sum = df["CondicionPago"].value_counts()
prod_sum = df.groupby("Producto")["Monto"].sum().sort_values(ascending=False).head(8)
region_sum = df.groupby("Region")["Monto"].sum().reindex(regiones)
vendedor_sum = df.groupby("Vendedor")["Monto"].sum().sort_values(ascending=False)
margen_vend = df.groupby("Vendedor")["%Margen"].mean().sort_values(ascending=False)
cumpl_vend = df.groupby("Vendedor")["%CumplimientoMeta"].mean().sort_values(ascending=False)

# --- Cálculos y Modelos para Hoja 2 (Estadísticos & ML) ---
monto_hist_counts, monto_hist_bins = np.histogram(df["Monto"], bins=25)
data_by_region = [df[df["Region"]==r]["Monto"].values for r in regiones]
corr = df[["Monto","Costo","Margen","%Margen","%CumplimientoMeta"]].corr()

# Regresión Lineal Multivariable
df_ml = df.copy()
le_region = LabelEncoder()
df_ml["Region_enc"] = le_region.fit_transform(df_ml["Region"])
X_reg = df_ml[["Mes","Region_enc"]]
y_reg = df_ml["Monto"]
model_reg = LinearRegression().fit(X_reg, y_reg)
meses_num = np.arange(1, 13)
reg_este_val = le_region.transform(["Este"])[0]
X_pred = np.column_stack([meses_num, np.repeat(reg_este_val, len(meses_num))])
y_pred_este = model_reg.predict(X_pred)

# Random Forest Classifier
le_cond = LabelEncoder()
df_ml["Cond_enc"] = le_cond.fit_transform(df_ml["CondicionPago"])
X_clf = df_ml[["Monto","Mes","Region_enc"]]
y_clf = df_ml["Cond_enc"]
X_train, X_test, y_train, y_test = train_test_split(
    X_clf, y_clf, test_size=0.25, random_state=42, stratify=y_clf
)
clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
y_pred_clf = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred_clf)

# K-Means Clustering sobre perfiles de Clientes
cliente_agg = df.groupby("Cliente").agg(
    total_gastado=("Monto","sum"),
    frecuencia=("Monto","count"),
    promedio=("Monto","mean"),
    margen_promedio=("%Margen","mean")
).reset_index()
scaler = StandardScaler()
X_cluster = scaler.fit_transform(cliente_agg[["total_gastado","frecuencia","promedio","margen_promedio"]])
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_cluster)
cliente_agg["cluster"] = kmeans.labels_
cluster_centers_orig = scaler.inverse_transform(kmeans.cluster_centers_)

# --- Cálculos y Modelos para Hoja 3 (Metas & Cumplimiento) ---
vend_meta = df.groupby("Vendedor")[["Meta_Vendedor","Monto"]].sum()
cumpl_hist_counts, cumpl_hist_bins = np.histogram(df["%CumplimientoMeta"], bins=20)
cumpl_region = df.groupby("Region")["%CumplimientoMeta"].mean().reindex(regiones)
cumpl_cat = df.groupby("Categoria")["%CumplimientoMeta"].mean().sort_values(ascending=False)
heat_data = (
    df.groupby(["Mes", "Region"])["%CumplimientoMeta"]
      .mean()
      .unstack()
      .reindex(range(1,13))
)

print(f"   [OK] Modelos entrenados exitosamente. Iniciando renderizado de animaciones...")

# ==============================================================================
# FASE 2: RENDERIZADO ANIMADO - HOJA 1: KPIS COMERCIALES (MATRIZ 4x2)
# ==============================================================================
print(f"\n[2/5] Renderizando Hoja 1: KPIs Comerciales ({NUM_FRAMES} frames)...")
frames_h1 = []

for f in range(NUM_FRAMES):
    prog = f / (NUM_FRAMES - 1)
    scale_b = min(1.0, prog * 1.25)

    fig = plt.figure(figsize=(16.0, 9.0), facecolor="#f8fafc", dpi=100)

    # Cabecera Ejecutiva Dinámica
    fig.text(0.045, 0.955, "HOJA 1 | TABLERO DE KPIS COMERCIALES & FACTURACIÓN (4×2)",
             fontsize=15.5, fontweight="bold", color="#0f172a")
    fase_txt = "Simulación en Vivo: Desplegando Ventas & Distribuciones" if prog < 0.85 else "Pipeline Convergido · Storytelling Activo"
    fig.text(0.045, 0.932, f"Estado Dinámico: {fase_txt} [{int(prog * 100)}%]",
             fontsize=10.0, color="#0284c7", fontweight="bold")
    fig.text(0.805, 0.948, "Arquitectura: Ing. Juancito Peña", fontsize=9.2, fontweight="bold", color="#1e293b",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="#e2e8f0", edgecolor="#cbd5e1"))

    gs1 = gridspec.GridSpec(2, 4, figure=fig, left=0.055, right=0.970, top=0.900, bottom=0.080, wspace=0.28, hspace=0.38)

    # 1. Ventas Mensuales por Año (Line Chart Multiserie)
    ax1 = fig.add_subplot(gs1[0, 0], facecolor="#ffffff")
    n_pts = max(2, int(12 * min(1.0, prog * 1.2)))
    for año, col in zip(pivot_ym.columns, ["#3b82f6", "#10b981"]):
        serie = pivot_ym[año]
        ax1.plot(pivot_ym.index[:n_pts], serie.iloc[:n_pts]/1000, marker="o", linewidth=2.0, color=col, label=f"Año {año}", markersize=5)
    ax1.set_xlim(1, 12)
    ax1.set_ylim(0, max(pivot_ym.max())/1000 * 1.18)
    ax1.set_xticks(range(1, 13))
    ax1.set_xticklabels(meses_abrev, fontsize=7.5)
    ax1.set_title("1. Ventas Mensuales por Año", fontsize=10.5, fontweight="bold", color="#0f172a")
    ax1.set_ylabel("Monto (Miles RD$)", fontsize=8, color="#475569")
    ax1.grid(True, linestyle=":", alpha=0.6, color="#cbd5e1")
    ax1.legend(loc="upper left", fontsize=7.5, frameon=True, facecolor="#ffffff")
    if prog > 0.6:
        ax1.text(0.04, 0.10, "• Crecimiento sostenido\n  en periodo 2025-2026", transform=ax1.transAxes,
                 fontsize=7.2, bbox=dict(boxstyle="round,pad=0.25", facecolor="#eff6ff", edgecolor="#bfdbfe"))

    # 2. Distribución Condición de Pago (Pie Chart)
    ax2 = fig.add_subplot(gs1[0, 1], facecolor="#ffffff")
    pie_colors = ["#2ecc71", "#e74c3c", "#3498db", "#f39c12"]
    # Animación de rotación y despliegue
    start_ang = prog * 360
    wedges, texts, autotexts = ax2.pie(
        cond_sum.values,
        labels=cond_sum.index if prog > 0.3 else None,
        autopct=lambda pct: f"{pct:.1f}%" if prog > 0.4 else "",
        colors=pie_colors,
        startangle=start_ang,
        textprops=dict(fontsize=8, fontweight="bold")
    )
    for at in autotexts:
        at.set_fontsize(7.5)
    ax2.set_title("2. Condición de Pago", fontsize=10.5, fontweight="bold", color="#0f172a")

    # 3. Top 8 Productos (Bar Chart con K)
    ax3 = fig.add_subplot(gs1[0, 2], facecolor="#ffffff")
    x_prod = np.arange(len(prod_sum))
    vals_prod_anim = (prod_sum.values * scale_b) / 1000
    bars3 = ax3.bar(x_prod, vals_prod_anim, color="#3498db", edgecolor="#1d4ed8", linewidth=0.6)
    ax3.set_xticks(x_prod)
    ax3.set_xticklabels(prod_sum.index, rotation=35, ha="right", fontsize=7.2)
    ax3.set_ylim(0, max(prod_sum.values)/1000 * 1.25)
    ax3.set_title("3. Top 8 Productos", fontsize=10.5, fontweight="bold", color="#0f172a")
    ax3.set_ylabel("Monto (k)", fontsize=8, color="#475569")
    ax3.grid(True, linestyle=":", alpha=0.5, color="#cbd5e1", axis="y")
    if prog > 0.7:
        for b, v_orig in zip(bars3, prod_sum.values):
            h = b.get_height()
            ax3.text(b.get_x() + b.get_width()/2, h + 2, f"{v_orig/1000:.1f}K", ha="center", va="bottom", fontsize=7, fontweight="bold")

    # 4. Ventas por Región (Bar Chart)
    ax4 = fig.add_subplot(gs1[0, 3], facecolor="#ffffff")
    x_reg_idx = np.arange(len(region_sum))
    vals_reg_anim = (region_sum.values * scale_b) / 1000
    bars4 = ax4.bar(x_reg_idx, vals_reg_anim, color="#27ae60", edgecolor="#166534", linewidth=0.6)
    ax4.set_xticks(x_reg_idx)
    ax4.set_xticklabels(region_sum.index, fontsize=8)
    ax4.set_ylim(0, max(region_sum.values)/1000 * 1.25)
    ax4.set_title("4. Ventas por Región", fontsize=10.5, fontweight="bold", color="#0f172a")
    ax4.set_ylabel("Monto (k)", fontsize=8, color="#475569")
    ax4.grid(True, linestyle=":", alpha=0.5, color="#cbd5e1", axis="y")
    if prog > 0.7:
        for b, v_orig in zip(bars4, region_sum.values):
            h = b.get_height()
            ax4.text(b.get_x() + b.get_width()/2, h + 2, f"{v_orig/1000:.1f}K", ha="center", va="bottom", fontsize=7, fontweight="bold")

    # 5. Ventas por Vendedor (Bar Chart Púrpura)
    ax5 = fig.add_subplot(gs1[1, 0], facecolor="#ffffff")
    x_vend = np.arange(len(vendedor_sum))
    vals_vend_anim = (vendedor_sum.values * scale_b) / 1000
    bars5 = ax5.bar(x_vend, vals_vend_anim, color="#9b59b6", edgecolor="#6b21a8", linewidth=0.6)
    ax5.set_xticks(x_vend)
    ax5.set_xticklabels(vendedor_sum.index, fontsize=8)
    ax5.set_ylim(0, max(vendedor_sum.values)/1000 * 1.25)
    ax5.set_title("5. Ventas por Vendedor", fontsize=10.5, fontweight="bold", color="#0f172a")
    ax5.set_ylabel("Monto (k)", fontsize=8, color="#475569")
    ax5.grid(True, linestyle=":", alpha=0.5, color="#cbd5e1", axis="y")
    if prog > 0.7:
        for b, v_orig in zip(bars5, vendedor_sum.values):
            h = b.get_height()
            ax5.text(b.get_x() + b.get_width()/2, h + 2, f"{v_orig/1000:.1f}K", ha="center", va="bottom", fontsize=7, fontweight="bold")

    # 6. % Margen Promedio por Vendedor (Bar Chart Naranja)
    ax6 = fig.add_subplot(gs1[1, 1], facecolor="#ffffff")
    x_mrg = np.arange(len(margen_vend))
    vals_mrg_anim = margen_vend.values * scale_b
    bars6 = ax6.bar(x_mrg, vals_mrg_anim, color="#e67e22", edgecolor="#c2410c", linewidth=0.6)
    ax6.set_xticks(x_mrg)
    ax6.set_xticklabels(margen_vend.index, rotation=35, ha="right", fontsize=7.5)
    ax6.set_ylim(0, max(margen_vend.values) * 1.25)
    ax6.set_title("6. % Margen Promedio", fontsize=10.5, fontweight="bold", color="#0f172a")
    ax6.set_ylabel("% Margen", fontsize=8, color="#475569")
    ax6.grid(True, linestyle=":", alpha=0.5, color="#cbd5e1", axis="y")
    if prog > 0.7:
        for b, v_orig in zip(bars6, margen_vend.values):
            h = b.get_height()
            ax6.text(b.get_x() + b.get_width()/2, h + 0.8, f"{v_orig:.1f}%", ha="center", va="bottom", fontsize=7, fontweight="bold")

    # 7. % Cumplimiento Meta por Vendedor (Bar Chart Morado)
    ax7 = fig.add_subplot(gs1[1, 2], facecolor="#ffffff")
    x_cump = np.arange(len(cumpl_vend))
    vals_cump_anim = cumpl_vend.values * scale_b
    bars7 = ax7.bar(x_cump, vals_cump_anim, color="purple", edgecolor="#4a044e", linewidth=0.6)
    ax7.axhline(100.0, color="#ef4444", linestyle="--", linewidth=1.2, label="Meta 100%")
    ax7.set_xticks(x_cump)
    ax7.set_xticklabels(cumpl_vend.index, rotation=35, ha="right", fontsize=7.5)
    ax7.set_ylim(0, max(cumpl_vend.values) * 1.25)
    ax7.set_title("7. % Cumplimiento Meta", fontsize=10.5, fontweight="bold", color="#0f172a")
    ax7.set_ylabel("% Cumplimiento", fontsize=8, color="#475569")
    ax7.grid(True, linestyle=":", alpha=0.5, color="#cbd5e1", axis="y")
    ax7.legend(loc="upper left", fontsize=7, frameon=True, facecolor="#ffffff")
    if prog > 0.7:
        for b, v_orig in zip(bars7, cumpl_vend.values):
            h = b.get_height()
            ax7.text(b.get_x() + b.get_width()/2, h + 1.2, f"{v_orig:.1f}%", ha="center", va="bottom", fontsize=7, fontweight="bold")

    # 8. Participación Territorial en Donut Chart
    ax8 = fig.add_subplot(gs1[1, 3], facecolor="#ffffff")
    donut_colors = ["#1abc9c", "#9b59b6", "#34495e", "#e67e22"]
    wedges8, texts8, autotexts8 = ax8.pie(
        region_sum.values,
        labels=region_sum.index if prog > 0.3 else None,
        autopct=lambda pct: f"{pct:.1f}%" if prog > 0.4 else "",
        colors=donut_colors,
        startangle=start_ang,
        textprops=dict(fontsize=8, fontweight="bold")
    )
    for at in autotexts8:
        at.set_fontsize(7.5)
    centre_circle = plt.Circle((0,0), 0.65, fc="white")
    ax8.add_artist(centre_circle)
    ax8.set_title("8. Mix Territorial", fontsize=10.5, fontweight="bold", color="#0f172a")

    # Guardar frame estático PNG en la última iteración
    if f == NUM_FRAMES - 1:
        plt.savefig(H1_PNG, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")

    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor=fig.get_facecolor(), edgecolor="none")
    buf.seek(0)
    frames_h1.append(Image.open(buf).copy())
    buf.close()
    plt.close(fig)

# Congelar último frame para lectura directiva
for _ in range(8):
    frames_h1.append(frames_h1[-1])

frames_h1[0].save(H1_GIF, save_all=True, append_images=frames_h1[1:], duration=120, loop=0, optimize=True)
print(f"   [OK] Hoja 1 generada: {H1_GIF} ({os.path.getsize(H1_GIF)/(1024*1024):.2f} MB)")

# ==============================================================================
# FASE 3: RENDERIZADO ANIMADO - HOJA 2: ESTADÍSTICOS & MACHINE LEARNING (3x2)
# ==============================================================================
print(f"\n[3/5] Renderizando Hoja 2: Estadísticos & Machine Learning ({NUM_FRAMES} frames)...")
frames_h2 = []

for f in range(NUM_FRAMES):
    prog = f / (NUM_FRAMES - 1)
    scale_b = min(1.0, prog * 1.25)

    fig = plt.figure(figsize=(16.0, 9.0), facecolor="#f8fafc", dpi=100)

    # Header
    fig.text(0.045, 0.955, "HOJA 2 | ESTADÍSTICOS DESCRIPTIVOS & MACHINE LEARNING (3×2)",
             fontsize=15.5, fontweight="bold", color="#0f172a")
    fase_txt = "Entrenando Modelos & Minería de Patrones..." if prog < 0.85 else "Modelos Convergidos: Random Forest & K-Means"
    fig.text(0.045, 0.932, f"Simulación Algorítmica: {fase_txt} [{int(prog * 100)}%]",
             fontsize=10.0, color="#8b5cf6", fontweight="bold")
    fig.text(0.805, 0.948, "Arquitectura: Ing. Juancito Peña", fontsize=9.2, fontweight="bold", color="#1e293b",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="#e2e8f0", edgecolor="#cbd5e1"))

    gs2 = gridspec.GridSpec(2, 3, figure=fig, left=0.065, right=0.965, top=0.900, bottom=0.080, wspace=0.28, hspace=0.38)

    # 1. Histograma de Montos
    ax1 = fig.add_subplot(gs2[0, 0], facecolor="#ffffff")
    counts_anim = monto_hist_counts * scale_b
    width_bar = (monto_hist_bins[1] - monto_hist_bins[0]) * 0.92
    for idx_b in range(len(counts_anim)):
        ax1.bar(monto_hist_bins[idx_b]/1000, counts_anim[idx_b], width=width_bar/1000,
                color="#3498db", edgecolor="#1d4ed8", alpha=0.85, linewidth=0.6)
    ax1.set_xlim(monto_hist_bins[0]/1000 - 1, monto_hist_bins[-1]/1000 + 1)
    ax1.set_ylim(0, max(monto_hist_counts) * 1.18)
    ax1.set_title("1. Histograma de Montos", fontsize=11, fontweight="bold", color="#0f172a")
    ax1.set_xlabel("Monto (Miles RD$)", fontsize=8.5, color="#475569")
    ax1.set_ylabel("Frecuencia", fontsize=8.5, color="#475569")
    ax1.grid(True, linestyle=":", alpha=0.5, color="#cbd5e1")
    if prog > 0.6:
        media_val = df["Monto"].mean() / 1000
        ax1.axvline(media_val, color="#e11d48", linestyle="--", linewidth=1.6, label=f"Media: {media_val:.1f}K")
        ax1.legend(loc="upper right", fontsize=7.5, frameon=True, facecolor="#ffffff")

    # 2. Boxplot por Región
    ax2 = fig.add_subplot(gs2[0, 1], facecolor="#ffffff")
    if prog > 0.2:
        # Dibujar boxplots dinámicos
        sub_data = [d[:max(10, int(len(d) * prog))]/1000 for d in data_by_region]
        bp = ax2.boxplot(sub_data, patch_artist=True, tick_labels=regiones,
                         boxprops=dict(facecolor="#f0fdf4", color="#16a34a", linewidth=1.2),
                         medianprops=dict(color="#dc2626", linewidth=1.8),
                         whiskerprops=dict(color="#16a34a", linewidth=1.2),
                         capprops=dict(color="#16a34a", linewidth=1.2),
                         flierprops=dict(marker='o', markersize=4, markerfacecolor='#e11d48', markeredgecolor='none'))
    ax2.set_title("2. Boxplot de Montos por Región", fontsize=11, fontweight="bold", color="#0f172a")
    ax2.set_ylabel("Monto (Miles RD$)", fontsize=8.5, color="#475569")
    ax2.grid(True, linestyle=":", alpha=0.5, color="#cbd5e1")

    # 3. Matriz de Correlaciones de Pearson (Heatmap)
    ax3 = fig.add_subplot(gs2[0, 2], facecolor="#ffffff")
    corr_anim = corr.values * scale_b
    im_c = ax3.imshow(corr_anim, cmap="coolwarm", vmin=-1, vmax=1)
    ax3.set_xticks(range(len(corr.columns)))
    ax3.set_xticklabels(corr.columns, rotation=35, ha="right", fontsize=7.5)
    ax3.set_yticks(range(len(corr.columns)))
    ax3.set_yticklabels(corr.columns, fontsize=7.5)
    for i_c in range(len(corr.columns)):
        for j_c in range(len(corr.columns)):
            v_c = corr.iloc[i_c, j_c] * scale_b
            ax3.text(j_c, i_c, f"{v_c:.2f}", ha="center", va="center", fontsize=7.5,
                     color="white" if abs(v_c) > 0.55 else "black", fontweight="bold")
    ax3.set_title("3. Correlaciones Financieras", fontsize=11, fontweight="bold", color="#0f172a")
    fig.colorbar(im_c, ax=ax3, fraction=0.046, pad=0.04)

    # 4. Regresión Lineal Multivariable (Mes + Región)
    ax4 = fig.add_subplot(gs2[1, 0], facecolor="#ffffff")
    n_scat = max(20, int(len(df_ml) * min(1.0, prog * 1.2)))
    ax4.scatter(df_ml["Mes"].iloc[:n_scat], df_ml["Monto"].iloc[:n_scat]/1000,
                alpha=0.25, color="gray", s=18, label="Datos Históricos")
    n_reg = max(2, int(12 * min(1.0, prog * 1.3)))
    ax4.plot(meses_num[:n_reg], y_pred_este[:n_reg]/1000, marker="o", color="#dc2626", linewidth=2.2, label="Predicción Región Este")
    ax4.set_xlim(1, 12)
    ax4.set_xticks(range(1, 13))
    ax4.set_xticklabels(meses_abrev, fontsize=7.5)
    ax4.set_title("4. Regresión Lineal (Mes + Región)", fontsize=11, fontweight="bold", color="#0f172a")
    ax4.set_xlabel("Mes", fontsize=8.5, color="#475569")
    ax4.set_ylabel("Monto (Miles RD$)", fontsize=8.5, color="#475569")
    ax4.grid(True, linestyle=":", alpha=0.5, color="#cbd5e1")
    ax4.legend(loc="upper left", fontsize=7.2, frameon=True, facecolor="#ffffff")
    if prog > 0.7:
        ax4.text(0.04, 0.08, f"• Tendencia estable\n  proyectada para Este", transform=ax4.transAxes,
                 fontsize=7.2, bbox=dict(boxstyle="round,pad=0.25", facecolor="#fef2f2", edgecolor="#fecaca"))

    # 5. Matriz de Confusión (Random Forest)
    ax5 = fig.add_subplot(gs2[1, 1], facecolor="#ffffff")
    cm_anim = (cm * scale_b).astype(int)
    im_cm = ax5.imshow(cm_anim, cmap="Blues", vmin=0, vmax=cm.max())
    ax5.set_xticks(range(len(le_cond.classes_)))
    ax5.set_xticklabels(le_cond.classes_, rotation=30, ha="right", fontsize=7.5)
    ax5.set_yticks(range(len(le_cond.classes_)))
    ax5.set_yticklabels(le_cond.classes_, fontsize=7.5)
    for i_cm in range(cm.shape[0]):
        for j_cm in range(cm.shape[1]):
            val_cm = cm_anim[i_cm, j_cm]
            ax5.text(j_cm, i_cm, str(val_cm), ha="center", va="center",
                     color="white" if val_cm > cm.max() * 0.55 else "black", fontsize=8, fontweight="bold")
    ax5.set_title("5. Matriz de Confusión (Random Forest)", fontsize=11, fontweight="bold", color="#0f172a")
    fig.colorbar(im_cm, ax=ax5, fraction=0.046, pad=0.04)

    # 6. Clustering de Clientes (K-Means)
    ax6 = fig.add_subplot(gs2[1, 2], facecolor="#ffffff")
    cl_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for cl in sorted(cliente_agg["cluster"].unique()):
        sub_c = cliente_agg[cliente_agg["cluster"] == cl]
        n_c = max(1, int(len(sub_c) * min(1.0, prog * 1.3)))
        ax6.scatter(sub_c["frecuencia"].iloc[:n_c], sub_c["total_gastado"].iloc[:n_c]/1000,
                    s=70, color=cl_colors[cl], edgecolors="white", linewidth=0.7, label=f"Cluster {cl}")
    if prog > 0.6:
        # Centros escalados
        ax6.scatter([c[1] for c in cluster_centers_orig], [c[0]/1000 for c in cluster_centers_orig],
                    marker="*", s=220, color="#dc2626", edgecolors="white", linewidth=1.2, zorder=6, label="Centroides")
    ax6.set_title("6. Clustering K-Means (Segmentación)", fontsize=11, fontweight="bold", color="#0f172a")
    ax6.set_xlabel("Frecuencia (Transacciones)", fontsize=8.5, color="#475569")
    ax6.set_ylabel("Total Gastado (Miles RD$)", fontsize=8.5, color="#475569")
    ax6.grid(True, linestyle=":", alpha=0.5, color="#cbd5e1")
    ax6.legend(loc="lower right", fontsize=7.2, frameon=True, facecolor="#ffffff")

    if f == NUM_FRAMES - 1:
        plt.savefig(H2_PNG, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")

    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor=fig.get_facecolor(), edgecolor="none")
    buf.seek(0)
    frames_h2.append(Image.open(buf).copy())
    buf.close()
    plt.close(fig)

for _ in range(8):
    frames_h2.append(frames_h2[-1])

frames_h2[0].save(H2_GIF, save_all=True, append_images=frames_h2[1:], duration=120, loop=0, optimize=True)
print(f"   [OK] Hoja 2 generada: {H2_GIF} ({os.path.getsize(H2_GIF)/(1024*1024):.2f} MB)")

# ==============================================================================
# FASE 4: RENDERIZADO ANIMADO - HOJA 3: METAS & CUMPLIMIENTO (3x2)
# ==============================================================================
print(f"\n[4/5] Renderizando Hoja 3: Metas & Cumplimiento ({NUM_FRAMES} frames)...")
frames_h3 = []

for f in range(NUM_FRAMES):
    prog = f / (NUM_FRAMES - 1)
    scale_b = min(1.0, prog * 1.25)

    fig = plt.figure(figsize=(16.0, 9.0), facecolor="#f8fafc", dpi=100)

    # Header
    fig.text(0.045, 0.955, "HOJA 3 | EVALUACIÓN DE METAS, CUMPLIMIENTO & RENTABILIDAD (3×2)",
             fontsize=15.5, fontweight="bold", color="#0f172a")
    fase_txt = "Simulación Dinámica: Control de Cuotas & Correlación de Rentabilidad" if prog < 0.85 else "Auditoría de Metas Finalizada · 100% Cobertura"
    fig.text(0.045, 0.932, f"Monitoreo de Cuotas: {fase_txt} [{int(prog * 100)}%]",
             fontsize=10.0, color="#f59e0b", fontweight="bold")
    fig.text(0.805, 0.948, "Arquitectura: Ing. Juancito Peña", fontsize=9.2, fontweight="bold", color="#1e293b",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="#e2e8f0", edgecolor="#cbd5e1"))

    gs3 = gridspec.GridSpec(2, 3, figure=fig, left=0.065, right=0.965, top=0.900, bottom=0.080, wspace=0.28, hspace=0.38)

    # 1. Barras Dobles: Meta vs Monto por Vendedor
    ax1 = fig.add_subplot(gs3[0, 0], facecolor="#ffffff")
    x_vm = np.arange(len(vend_meta.index))
    width_vm = 0.35
    v_meta_anim = (vend_meta["Meta_Vendedor"].values * scale_b) / 1000
    v_monto_anim = (vend_meta["Monto"].values * scale_b) / 1000
    b_meta = ax1.bar(x_vm - width_vm/2, v_meta_anim, width_vm, label="Meta Asignada", color="#94a3b8", edgecolor="#64748b", linewidth=0.6)
    b_monto = ax1.bar(x_vm + width_vm/2, v_monto_anim, width_vm, label="Monto Facturado", color="#3498db", edgecolor="#1d4ed8", linewidth=0.6)
    ax1.set_xticks(x_vm)
    ax1.set_xticklabels(vend_meta.index, rotation=30, ha="right", fontsize=8)
    ax1.set_ylim(0, max(vend_meta.max())/1000 * 1.25)
    ax1.set_title("1. Meta vs. Monto por Vendedor", fontsize=11, fontweight="bold", color="#0f172a")
    ax1.set_ylabel("Monto (Miles RD$)", fontsize=8.5, color="#475569")
    ax1.grid(True, linestyle=":", alpha=0.5, color="#cbd5e1", axis="y")
    ax1.legend(loc="upper left", fontsize=7.5, frameon=True, facecolor="#ffffff")

    # 2. Histograma de % Cumplimiento
    ax2 = fig.add_subplot(gs3[0, 1], facecolor="#ffffff")
    cump_counts_anim = cumpl_hist_counts * scale_b
    w_ch = (cumpl_hist_bins[1] - cumpl_hist_bins[0]) * 0.92
    for idx_c in range(len(cump_counts_anim)):
        ax2.bar(cumpl_hist_bins[idx_c], cump_counts_anim[idx_c], width=w_ch,
                color="#e74c3c", edgecolor="#991b1b", alpha=0.85, linewidth=0.6)
    ax2.axvline(100.0, color="#16a34a", linestyle="--", linewidth=1.6, label="Meta 100%")
    ax2.set_xlim(cumpl_hist_bins[0] - 2, cumpl_hist_bins[-1] + 2)
    ax2.set_ylim(0, max(cumpl_hist_counts) * 1.18)
    ax2.set_title("2. Distribución % Cumplimiento de Metas", fontsize=11, fontweight="bold", color="#0f172a")
    ax2.set_xlabel("% Cumplimiento", fontsize=8.5, color="#475569")
    ax2.set_ylabel("Frecuencia", fontsize=8.5, color="#475569")
    ax2.grid(True, linestyle=":", alpha=0.5, color="#cbd5e1")
    ax2.legend(loc="upper left", fontsize=7.5, frameon=True, facecolor="#ffffff")

    # 3. % Cumplimiento por Región
    ax3 = fig.add_subplot(gs3[0, 2], facecolor="#ffffff")
    x_cr = np.arange(len(cumpl_region))
    v_cr_anim = cumpl_region.values * scale_b
    bars_cr = ax3.bar(x_cr, v_cr_anim, color="#2ecc71", edgecolor="#15803d", linewidth=0.6)
    ax3.axhline(100.0, color="#ef4444", linestyle="--", linewidth=1.2, label="Meta 100%")
    ax3.set_xticks(x_cr)
    ax3.set_xticklabels(cumpl_region.index, fontsize=8)
    ax3.set_ylim(0, max(cumpl_region.values) * 1.25)
    ax3.set_title("3. % Cumplimiento por Región", fontsize=11, fontweight="bold", color="#0f172a")
    ax3.set_ylabel("% Cumplimiento", fontsize=8.5, color="#475569")
    ax3.grid(True, linestyle=":", alpha=0.5, color="#cbd5e1", axis="y")
    ax3.legend(loc="upper left", fontsize=7.2, frameon=True, facecolor="#ffffff")
    if prog > 0.7:
        for b, v_orig in zip(bars_cr, cumpl_region.values):
            h = b.get_height()
            ax3.text(b.get_x() + b.get_width()/2, h + 1.2, f"{v_orig:.1f}%", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    # 4. % Cumplimiento por Categoría
    ax4 = fig.add_subplot(gs3[1, 0], facecolor="#ffffff")
    x_cc = np.arange(len(cumpl_cat))
    v_cc_anim = cumpl_cat.values * scale_b
    bars_cc = ax4.bar(x_cc, v_cc_anim, color="#9b59b6", edgecolor="#6b21a8", linewidth=0.6)
    ax4.axhline(100.0, color="#ef4444", linestyle="--", linewidth=1.2, label="Meta 100%")
    ax4.set_xticks(x_cc)
    ax4.set_xticklabels(cumpl_cat.index, rotation=30, ha="right", fontsize=7.5)
    ax4.set_ylim(0, max(cumpl_cat.values) * 1.25)
    ax4.set_title("4. % Cumplimiento por Categoría", fontsize=11, fontweight="bold", color="#0f172a")
    ax4.set_ylabel("% Cumplimiento", fontsize=8.5, color="#475569")
    ax4.grid(True, linestyle=":", alpha=0.5, color="#cbd5e1", axis="y")
    ax4.legend(loc="upper left", fontsize=7.2, frameon=True, facecolor="#ffffff")
    if prog > 0.7:
        for b, v_orig in zip(bars_cc, cumpl_cat.values):
            h = b.get_height()
            ax4.text(b.get_x() + b.get_width()/2, h + 1.2, f"{v_orig:.1f}%", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    # 5. Scatter % Margen vs % CumplimientoMeta
    ax5 = fig.add_subplot(gs3[1, 1], facecolor="#ffffff")
    n_pts_scat = max(25, int(len(df) * min(1.0, prog * 1.25)))
    ax5.scatter(df["%Margen"].iloc[:n_pts_scat], df["%CumplimientoMeta"].iloc[:n_pts_scat],
                alpha=0.45, color="#34495e", edgecolors="none", s=22)
    ax5.axvline(df["%Margen"].mean(), color="#e67e22", linestyle=":", linewidth=1.2, label="Media % Margen")
    ax5.axhline(100.0, color="#ef4444", linestyle="--", linewidth=1.2, label="Meta 100%")
    ax5.set_title("5. % Margen vs. % Cumplimiento", fontsize=11, fontweight="bold", color="#0f172a")
    ax5.set_xlabel("% Margen", fontsize=8.5, color="#475569")
    ax5.set_ylabel("% Cumplimiento", fontsize=8.5, color="#475569")
    ax5.grid(True, linestyle=":", alpha=0.5, color="#cbd5e1")
    ax5.legend(loc="upper right", fontsize=7.2, frameon=True, facecolor="#ffffff")

    # 6. Heatmap de Cumplimiento por Mes y Región
    ax6 = fig.add_subplot(gs3[1, 2], facecolor="#ffffff")
    heat_anim = heat_data.values * scale_b
    im_h = ax6.imshow(heat_anim, cmap="YlGnBu", aspect="auto", vmin=80, vmax=125)
    ax6.set_xticks(range(len(heat_data.columns)))
    ax6.set_xticklabels(heat_data.columns, fontsize=8, fontweight="bold")
    ax6.set_yticks(range(12))
    ax6.set_yticklabels(meses_abrev, fontsize=7.5)
    for i_h in range(len(heat_data.index)):
        for j_h in range(len(heat_data.columns)):
            val_h = heat_anim[i_h, j_h]
            if val_h > 0:
                ax6.text(j_h, i_h, f"{val_h:.0f}%", ha="center", va="center",
                         color="white" if val_h > 105 else "black", fontsize=7.2, fontweight="bold")
    ax6.set_title("6. Cumplimiento por Mes & Región", fontsize=11, fontweight="bold", color="#0f172a")
    cbar3 = fig.colorbar(im_h, ax=ax6, shrink=0.88, pad=0.04)
    cbar3.set_label("% Cumplimiento", fontsize=8, rotation=90)

    if f == NUM_FRAMES - 1:
        plt.savefig(H3_PNG, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")

    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor=fig.get_facecolor(), edgecolor="none")
    buf.seek(0)
    frames_h3.append(Image.open(buf).copy())
    buf.close()
    plt.close(fig)

for _ in range(8):
    frames_h3.append(frames_h3[-1])

frames_h3[0].save(H3_GIF, save_all=True, append_images=frames_h3[1:], duration=120, loop=0, optimize=True)
print(f"   [OK] Hoja 3 generada: {H3_GIF} ({os.path.getsize(H3_GIF)/(1024*1024):.2f} MB)")

# ==============================================================================
# FASE 5: COMPILACIÓN DE LA SUITE MAESTRA COMPLETA (HOJA 1 + HOJA 2 + HOJA 3)
# ==============================================================================
print(f"\n[5/5] Compilando Suite Maestra Todo-en-Uno (20 Gráficos Animados en un solo GIF)...")
frames_master = []

# Añadir animación de Hoja 1
frames_master.extend(frames_h1)

# Pausa intermedia
for _ in range(8):
    frames_master.append(frames_h1[-1])

# Añadir animación de Hoja 2
frames_master.extend(frames_h2)

# Pausa intermedia
for _ in range(8):
    frames_master.append(frames_h2[-1])

# Añadir animación de Hoja 3
frames_master.extend(frames_h3)

# Pausa final
for _ in range(10):
    frames_master.append(frames_h3[-1])

frames_master[0].save(
    MASTER_GIF,
    save_all=True,
    append_images=frames_master[1:],
    duration=120,
    loop=0,
    optimize=True
)

t_total = time.time() - t_inicio
master_mb = os.path.getsize(MASTER_GIF) / (1024 * 1024)

# Resumen Ejecutivo en Consola
resumen = {
    "Total Ventas (RD$)": f"${df['Monto'].sum():,.2f}",
    "Promedio Venta (RD$)": f"${df['Monto'].mean():,.2f}",
    "Venta Máxima (RD$)": f"${df['Monto'].max():,.2f}",
    "Venta Mínima (RD$)": f"${df['Monto'].min():,.2f}",
    "Productos Únicos": df["Producto"].nunique(),
    "Clientes Únicos": df["Cliente"].nunique(),
    "Vendedores Únicos": df["Vendedor"].nunique(),
    "Regiones Únicas": df["Region"].nunique(),
    "Condiciones de Pago": df["CondicionPago"].nunique(),
    "Total Transacciones": len(df)
}

print("\n" + "=" * 80)
print("[EXITO] PIPELINE MAESTRO 10 EJECUTADO Y FINALIZADO CON EXITO")
print(f"Tiempo Total de Procesamiento: {t_total:.1f} segundos")
print("=" * 80)
print("\nTABLA RESUMEN DEL INFORME:")
for k, v in resumen.items():
    print(f"  • {k:<25}: {v}")

print("-" * 80)
print(f"1. Hoja 1 Animada (.GIF):     {os.path.abspath(H1_GIF)} ({os.path.getsize(H1_GIF)/(1024*1024):.2f} MB)")
print(f"2. Hoja 1 Estática (.PNG):    {os.path.abspath(H1_PNG)} (300 DPI Ultra HD)")
print(f"3. Hoja 2 Animada (.GIF):     {os.path.abspath(H2_GIF)} ({os.path.getsize(H2_GIF)/(1024*1024):.2f} MB)")
print(f"4. Hoja 2 Estática (.PNG):    {os.path.abspath(H2_PNG)} (300 DPI Ultra HD)")
print(f"5. Hoja 3 Animada (.GIF):     {os.path.abspath(H3_GIF)} ({os.path.getsize(H3_GIF)/(1024*1024):.2f} MB)")
print(f"6. Hoja 3 Estática (.PNG):    {os.path.abspath(H3_PNG)} (300 DPI Ultra HD)")
print(f"7. SUITE TODO-EN-UNO (.GIF):  {os.path.abspath(MASTER_GIF)} ({master_mb:.2f} MB)")
print(f"8. Dataset Persistido (.XLSX):{os.path.abspath(EXCEL_OUTPUT)}")
print("=" * 80)
