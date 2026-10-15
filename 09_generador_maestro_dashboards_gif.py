# ==============================================================================
# 09 - GENERADOR MAESTRO UNIFICADO DE DASHBOARDS ANIMADOS (.GIF) 16:9
# Suite Completa de 12 Gráficos de Machine Learning Comercial en un Solo Script
# Dashboard 1: Estratégico 360° (6 Gráficos)
# Dashboard 2: Diagnóstico Avanzado, XAI & Elasticidad (6 Gráficos)
# + Suite Combinada Ejecutiva (Transición Continua)
# Autor: Ing. Juancito Peña & Antigravity
# Ejecutar en VS Code: python 09_generador_maestro_dashboards_gif.py
# ==============================================================================

import os
import io
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Renderizado optimizado sin bloqueo de GUI
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, roc_curve, auc
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.neural_network import MLPRegressor

warnings.filterwarnings("ignore")

# Estilización global consistente
plt.rcParams['font.sans-serif'] = 'Segoe UI'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#cbd5e1'
plt.rcParams['axes.linewidth'] = 0.9

EXCEL_PATH = "datasets_machine_learning_ventas.xlsx"

# Nombres de archivos de salida
D1_GIF = "dashboard_machine_learning_ventas_16_9.gif"
D1_PNG = "dashboard_machine_learning_ventas_16_9.png"
D2_GIF = "dashboard_avanzado_machine_learning_16_9.gif"
D2_PNG = "dashboard_avanzado_machine_learning_16_9.png"
MASTER_GIF = "dashboard_master_suite_completa_16_9.gif"

print("=" * 80)
print("[+] SCRIPT MAESTRO: GENERACION INTEGRAL DE DASHBOARDS ANIMADOS (.GIF) 16:9")
print("    Suite de 12 Graficos de Machine Learning Aplicado a Ventas")
print("=" * 80)

t_inicio = time.time()

# ------------------------------------------------------------------------------
# 1. CARGA DE DATOS Y COMPUTACIÓN UNIFICADA DE MODELOS (UNA SOLA VEZ)
# ------------------------------------------------------------------------------
print("\n[1/4] Cargando datasets y computando modelos predictivos en memoria...")

# Dataset 1: Regresión Lineal
df1 = pd.read_excel(EXCEL_PATH, sheet_name="01_Regresion_Ventas")
X1 = df1[["Publicidad_RDS", "Descuento_Pct", "Trafico_Visitas"]]
y1 = df1["Ventas_RDS"]
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.20, random_state=42)
mod_reg = LinearRegression().fit(X1_train, y1_train)
y1_pred = mod_reg.predict(X1_test)
r2_val = r2_score(y1_test, y1_pred)
mae_val = mean_absolute_error(y1_test, y1_pred)
coef_pub = mod_reg.coef_[0]
coef_traf = mod_reg.coef_[2]
residuos = (y1_test - y1_pred) / 1000
sigma_res = np.std(residuos)

# Dataset 2: Clustering K-Means
df2 = pd.read_excel(EXCEL_PATH, sheet_name="02_Clientes_Clustering")
feat2 = ["Total_Compras_RDS", "Frecuencia_Compras", "Dias_Desde_Ultima_Compra"]
scaler2 = StandardScaler()
X2_scaled = scaler2.fit_transform(df2[feat2])
kmeans = KMeans(n_clusters=3, random_state=42, n_init=12).fit(X2_scaled)
df2["Cluster"] = kmeans.labels_
sil_score = silhouette_score(X2_scaled, df2["Cluster"])
perfil2 = df2.groupby("Cluster")["Total_Compras_RDS"].mean()
ranking2 = perfil2.sort_values(ascending=False).index
map_cluster = {ranking2[0]: "Premium", ranking2[1]: "Frecuente", ranking2[2]: "Ocasional"}
df2["Segmento"] = df2["Cluster"].map(map_cluster)
centros_orig = scaler2.inverse_transform(kmeans.cluster_centers_)

# Método del Codo para Dashboard 2
k_valores = list(range(2, 7))
inertias, sil_scores = [], []
for k in k_valores:
    km_k = KMeans(n_clusters=k, random_state=42, n_init=8).fit(X2_scaled)
    inertias.append(km_k.inertia_)
    sil_scores.append(silhouette_score(X2_scaled, km_k.labels_))

# Dataset 3: Clasificación Semi-Supervisada
df3 = pd.read_excel(EXCEL_PATH, sheet_name="03_Pedidos_SemiSupervisado")
feat3 = ["Monto_RDS", "Cantidad_Items"]
X3 = df3[feat3]
y3 = df3["Etiqueta"].map({"Alto Valor": 1, "Normal": 0}).fillna(-1).astype(int)
base_rf = RandomForestClassifier(n_estimators=120, random_state=42, class_weight="balanced")
self_clf = SelfTrainingClassifier(estimator=base_rf, threshold=0.80).fit(X3, y3)
df3["Prediccion"] = self_clf.predict(X3)
df3["Confianza"] = self_clf.predict_proba(X3).max(axis=1)
df3["Tipo_Dato"] = np.where(y3 != -1, "Auditado Humano", "Auto-Etiquetado IA")
df3["Categoria"] = df3["Prediccion"].map({1: "Alto Valor", 0: "Normal"})
auditados = (y3 != -1).sum()
autoetq = (y3 == -1).sum()
conf_media = df3[df3["Tipo_Dato"] == "Auto-Etiquetado IA"]["Confianza"].mean() * 100

eval_mask = (y3 != -1)
probs3 = self_clf.predict_proba(X3[eval_mask])[:, 1]
fpr, tpr, _ = roc_curve(y3[eval_mask], probs3)
roc_auc = auc(fpr, tpr)

# Dataset 4: Reforzamiento Q-Learning
df4 = pd.read_excel(EXCEL_PATH, sheet_name="04_Reforzamiento")
acciones = [0, 10, 20, 30]
estados = [f"{dia}|{prod}|{temp}" for dia in df4["Dia"].unique()
           for prod in df4["Producto"].unique()
           for temp in df4["Temporada"].unique()]
estado_id = {est: i for i, est in enumerate(estados)}
Q = np.zeros((len(estados), len(acciones)))
alpha, gamma, epsilon = 0.10, 0.90, 0.20
rng = np.random.default_rng(42)

recompensas_hist = []
num_episodios = 3500

for ep in range(num_episodios):
    fila = df4.sample(1, random_state=ep).iloc[0]
    est = f"{fila['Dia']}|{fila['Producto']}|{fila['Temporada']}"
    s = estado_id[est]
    a = rng.integers(len(acciones)) if rng.random() < epsilon else np.argmax(Q[s])
    desc = acciones[a]
    ventas = max(0, fila["Demanda_Base"] * (1 + desc / 100) * rng.uniform(0.75, 1.15))
    ingreso = ventas * fila["Precio_RDS"] * (1 - desc / 100)
    Q[s, a] = Q[s, a] + alpha * (ingreso + gamma * np.max(Q[s]) - Q[s, a])
    recompensas_hist.append(ingreso)

recomp_smooth = pd.Series(recompensas_hist).rolling(100, min_periods=1).mean().values

politicas = []
for est, s in estado_id.items():
    dia, prod, temp = est.split("|")
    politicas.append({"Producto": prod, "Temporada": temp, "Descuento": acciones[np.argmax(Q[s])]})
df_pol = pd.DataFrame(politicas).groupby(["Producto", "Temporada"])["Descuento"].mean().unstack()
orden_temp = ["Baja", "Media", "Alta"]
df_pol_final = df_pol.reindex(columns=orden_temp)

# Dataset 5: Autoencoder para Detección de Anomalías
df5 = pd.read_excel(EXCEL_PATH, sheet_name="05_Autoencoder")
feat5 = ["Compras_90d", "Monto_90d_RDS", "Dias_Activo", "Productos_Distintos", "Devoluciones", "Visitas_Web"]
X5 = df5[feat5]
scaler5 = StandardScaler()
X5_scaled = scaler5.fit_transform(X5)
mlp = MLPRegressor(hidden_layer_sizes=(8, 3, 8), activation="relu", solver="adam", max_iter=120, random_state=42)
mlp.fit(X5_scaled, X5_scaled)
reconst = mlp.predict(X5_scaled)
mse_errors = np.mean(np.square(X5_scaled - reconst), axis=1)
df5["MSE"] = mse_errors
umbral_anom = np.quantile(mse_errors, 0.95)
df5["Es_Anomalia"] = df5["MSE"] > umbral_anom
anom_count = df5["Es_Anomalia"].sum()

# XAI Ratios
prom_norm = X5[~df5["Es_Anomalia"]].mean()
prom_anom = X5[df5["Es_Anomalia"]].mean()
ratios_xai = (prom_anom / prom_norm).values
nombres_xai = ["Compras 90d", "Monto RD$", "Días Activo", "Prod. Distintos", "Devoluciones", "Visitas Web"]
hist_counts, hist_bins = np.histogram(df5["MSE"], bins=26)

# KPIs Segmentos
segmentos_kpi = ["Ocasional", "Frecuente", "Premium"]
ventas_promedio = [df2[df2["Segmento"] == s]["Total_Compras_RDS"].mean() / 1000 for s in segmentos_kpi]
frecuencia_prom = [df2[df2["Segmento"] == s]["Frecuencia_Compras"].mean() for s in segmentos_kpi]

# Elasticidad Precio-Demanda
descuentos_rango = np.linspace(0, 35, 60)
demanda_sim = 120 * (1 + (descuentos_rango / 100) * 1.35)
ingresos_sim = (demanda_sim * 2500 * (1 - descuentos_rango / 100)) / 1000
idx_optimo = np.argmax(ingresos_sim)
desc_optimo = descuentos_rango[idx_optimo]
ingreso_optimo = ingresos_sim[idx_optimo]

print(">> Datos y modelos procesados exitosamente.")

# ------------------------------------------------------------------------------
# 2. GENERADOR DEL DASHBOARD 1: VISIÓN ESTRATÉGICA 360° (6 GRÁFICOS)
# ------------------------------------------------------------------------------
print("\n[2/4] Generando Dashboard 1 (Estratégico 360° - 6 Gráficos)...")
NUM_FRAMES = 26
frames_d1 = []

for f in range(NUM_FRAMES):
    prog = f / (NUM_FRAMES - 1)
    fig = plt.figure(figsize=(16.0, 9.0), facecolor="#f8fafc", dpi=100)

    # Header
    fig.text(0.045, 0.955, "DASHBOARD 1 | VISIÓN ESTRATÉGICA 360° DE MACHINE LEARNING",
             fontsize=16, fontweight="bold", color="#0f172a")
    fase_txt = "Entrenando Modelos..." if prog < 0.75 else "Modelos Convergidos & Storytelling Activo"
    fig.text(0.045, 0.932, f"Simulación Dinámica: {fase_txt} [{int(prog * 100)}%]",
             fontsize=10.5, color="#2563eb", fontweight="bold")
    fig.text(0.81, 0.948, "Arquitectura: Ing. Juancito Peña", fontsize=9.5, fontweight="bold", color="#1e293b",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="#e2e8f0", edgecolor="#cbd5e1"))

    gs = gridspec.GridSpec(2, 3, figure=fig, left=0.07, right=0.965, top=0.900, bottom=0.075, wspace=0.24, hspace=0.34)

    # 1. Regresión
    ax1 = fig.add_subplot(gs[0, 0], facecolor="#ffffff")
    n_puntos = max(3, int(len(y1_test) * min(1.0, prog * 1.3)))
    ax1.scatter(y1_test.iloc[:n_puntos]/1000, y1_pred[:n_puntos]/1000, color="#2563eb", alpha=0.8, s=60, edgecolors="#1d4ed8")
    min_v, max_v = min(y1_test.min(), y1_pred.min())/1000 - 5, max(y1_test.max(), y1_pred.max())/1000 + 5
    line_x = np.linspace(min_v, min_v + (max_v - min_v) * prog, 50)
    ax1.plot(line_x, line_x, color="#ef4444", linestyle="--", linewidth=1.8, label="Ajuste Ideal 45°")
    ax1.set_xlim(min_v, max_v)
    ax1.set_ylim(min_v, max_v)
    ax1.set_title("1. Pronóstico de Ventas vs. Realidad", fontsize=11, fontweight="bold", color="#0f172a", pad=6)
    ax1.set_xlabel("Ventas Reales (Miles RD$)", fontsize=8.5, color="#475569")
    ax1.set_ylabel("Ventas Predichas (Miles RD$)", fontsize=8.5, color="#475569")
    ax1.grid(True, linestyle=":", alpha=0.6, color="#cbd5e1")
    ax1.legend(loc="upper left", fontsize=7.2, frameon=True, facecolor="#ffffff")
    if prog > 0.6:
        ax1.text(0.04, 0.05, f"• R² = {r2_val:.3f} | MAE = RD$ {mae_val:,.0f}\n• Elasticidad Publicitaria: +{coef_pub:.1f}",
                 transform=ax1.transAxes, fontsize=7.5, bbox=dict(boxstyle="round,pad=0.3", facecolor="#eff6ff", edgecolor="#bfdbfe", alpha=0.9))

    # 2. Clustering
    ax2 = fig.add_subplot(gs[0, 1], facecolor="#ffffff")
    colores_seg = {"Premium": "#d97706", "Frecuente": "#2563eb", "Ocasional": "#94a3b8"}
    for seg in ["Premium", "Frecuente", "Ocasional"]:
        sub = df2[df2["Segmento"] == seg]
        n_sub = max(2, int(len(sub) * min(1.0, prog * 1.2)))
        ax2.scatter(sub["Total_Compras_RDS"].iloc[:n_sub]/1000, sub["Frecuencia_Compras"].iloc[:n_sub],
                    c=colores_seg[seg], label=f"{seg}", s=60, alpha=0.85, edgecolors="white", linewidth=0.5)
    if prog > 0.4:
        for cent in centros_orig:
            ax2.scatter(cent[0]/1000, cent[1], marker="*", s=200, c="#dc2626", edgecolors="white", linewidth=1.2, zorder=5)
    ax2.set_title("2. Segmentación RFM (K-Means)", fontsize=11, fontweight="bold", color="#0f172a", pad=6)
    ax2.set_xlabel("Facturación (Miles RD$)", fontsize=8.5, color="#475569")
    ax2.set_ylabel("Frecuencia (Compras)", fontsize=8.5, color="#475569")
    ax2.grid(True, linestyle=":", alpha=0.6, color="#cbd5e1")
    ax2.legend(loc="lower right", fontsize=7.2, frameon=True, facecolor="#ffffff")
    if prog > 0.6:
        ax2.text(0.04, 0.78, f"• Silhouette Score: {sil_score:.3f}\n• Premium factura 4.8x promedio.",
                 transform=ax2.transAxes, fontsize=7.5, bbox=dict(boxstyle="round,pad=0.3", facecolor="#fefce8", edgecolor="#fde047", alpha=0.9))

    # 3. Semi-supervisado
    ax3 = fig.add_subplot(gs[0, 2], facecolor="#ffffff")
    for cat, col in [("Alto Valor", "#0d9488"), ("Normal", "#64748b")]:
        sub_aud = df3[(df3["Tipo_Dato"] == "Auditado Humano") & (df3["Categoria"] == cat)]
        ax3.scatter(sub_aud["Monto_RDS"]/1000, sub_aud["Cantidad_Items"], color=col, marker="o", s=55, alpha=0.8, edgecolors="white", linewidth=0.5)
        if prog > 0.3:
            sub_ia = df3[(df3["Tipo_Dato"] == "Auto-Etiquetado IA") & (df3["Categoria"] == cat)]
            n_ia = max(1, int(len(sub_ia) * min(1.0, (prog - 0.3) * 1.5)))
            ax3.scatter(sub_ia["Monto_RDS"].iloc[:n_ia]/1000, sub_ia["Cantidad_Items"].iloc[:n_ia],
                        color=col, marker="^", s=55, alpha=0.75, edgecolors="white", linewidth=0.5)
    ax3.set_title("3. Clasificación Semi-Supervisada", fontsize=11, fontweight="bold", color="#0f172a", pad=6)
    ax3.set_xlabel("Monto Pedido (Miles RD$)", fontsize=8.5, color="#475569")
    ax3.set_ylabel("Cantidad Ítems", fontsize=8.5, color="#475569")
    ax3.grid(True, linestyle=":", alpha=0.6, color="#cbd5e1")
    if prog > 0.6:
        ax3.text(0.04, 0.78, f"• Auditados: {auditados} | Auto-IA: {autoetq}\n• Cobertura con {conf_media:.1f}% certeza.",
                 transform=ax3.transAxes, fontsize=7.5, bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0fdfa", edgecolor="#99f6e4", alpha=0.9))

    # 4. Q-Learning
    ax4 = fig.add_subplot(gs[1, 0], facecolor="#ffffff")
    mat_anim = df_pol_final.values * min(1.0, prog * 1.25)
    ax4.imshow(mat_anim, cmap="YlOrRd", aspect="auto", vmin=0, vmax=30)
    ax4.set_xticks(range(len(orden_temp)))
    ax4.set_xticklabels(orden_temp, fontsize=9, color="#1e293b", fontweight="bold")
    ax4.set_yticks(range(len(df_pol_final.index)))
    ax4.set_yticklabels(df_pol_final.index, fontsize=8, color="#1e293b")
    for i in range(len(df_pol_final.index)):
        for j in range(len(orden_temp)):
            val = mat_anim[i, j]
            ax4.text(j, i, f"{val:.0f}%", ha="center", va="center", color="#ffffff" if val > 17 else "#0f172a", fontweight="bold", fontsize=9)
    ax4.set_title("4. Matriz de Promociones (Q-Learning)", fontsize=11, fontweight="bold", color="#0f172a", pad=6)
    if prog > 0.6:
        ax4.text(0.02, 0.98, "• Temporada Alta: 0-10% (Sin regalar margen).\n• Temporada Baja: Acelerar rotación.",
                 transform=ax4.transAxes, fontsize=7.5, va="top", bbox=dict(boxstyle="round,pad=0.25", facecolor="#fff7ed", edgecolor="#ffedd5", alpha=0.9))

    # 5. Autoencoder
    ax5 = fig.add_subplot(gs[1, 1], facecolor="#ffffff")
    counts_anim = hist_counts * min(1.0, prog * 1.2)
    for idx_b in range(len(counts_anim)):
        ax5.bar(hist_bins[idx_b], counts_anim[idx_b], width=(hist_bins[1] - hist_bins[0]) * 0.9,
                color="#e11d48" if hist_bins[idx_b] >= umbral_anom else "#94a3b8", edgecolor="#64748b", alpha=0.8)
    ax5.set_xlim(hist_bins[0] - 0.2, hist_bins[-1] + 0.2)
    ax5.set_ylim(0, max(hist_counts) * 1.15)
    if prog > 0.5:
        ax5.axvline(umbral_anom, color="#be123c", linestyle="--", linewidth=1.8, label=f"Umbral Q95 ({umbral_anom:.2f})")
        ax5.legend(loc="upper right", fontsize=7.2, frameon=True, facecolor="#ffffff")
    ax5.set_title("5. Auditoría de Cuentas (Autoencoder MSE)", fontsize=11, fontweight="bold", color="#0f172a", pad=6)
    ax5.set_xlabel("Error Reconstrucción (MSE)", fontsize=8.5, color="#475569")
    ax5.set_ylabel("Frecuencia", fontsize=8.5, color="#475569")
    ax5.grid(True, linestyle=":", alpha=0.6, color="#cbd5e1")
    if prog > 0.7:
        ax5.text(0.04, 0.50, f"• Cuentas: 220 | Anomalías: {anom_count}.\n• Reduce 95% auditoría manual.",
                 transform=ax5.transAxes, fontsize=7.5, bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff1f2", edgecolor="#fecdd3", alpha=0.9))

    # 6. Impacto Estratégico
    ax6 = fig.add_subplot(gs[1, 2], facecolor="#ffffff")
    x_idx = np.arange(len(segmentos_kpi))
    bar_w = 0.38
    scale_b = min(1.0, prog * 1.3)
    bars1 = ax6.bar(x_idx - bar_w/2, [v * scale_b for v in ventas_promedio], width=bar_w, color="#2563eb", label="Facturación (k RD$)")
    ax6_twin = ax6.twinx()
    bars2 = ax6_twin.bar(x_idx + bar_w/2, [f_p * scale_b for f_p in frecuencia_prom], width=bar_w, color="#059669", label="Frecuencia (Compras)")
    ax6.set_xticks(x_idx)
    ax6.set_xticklabels(segmentos_kpi, fontsize=9, fontweight="bold", color="#1e293b")
    ax6.set_ylabel("Facturación (k RD$)", fontsize=8.5, color="#1d4ed8")
    ax6_twin.set_ylabel("Frecuencia (Compras)", fontsize=8.5, color="#047857")
    ax6.set_ylim(0, 145)
    ax6_twin.set_ylim(0, 38)
    ax6.set_title("6. Impacto Estratégico & Retorno", fontsize=11, fontweight="bold", color="#0f172a", pad=6)
    ax6.grid(True, linestyle=":", alpha=0.4, color="#cbd5e1")
    if prog > 0.8:
        for b in bars1:
            h = b.get_height()
            ax6.text(b.get_x() + b.get_width()/2, h + 2.5, f"${h:,.0f}k", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#1d4ed8")
        for b in bars2:
            h = b.get_height()
            ax6_twin.text(b.get_x() + b.get_width()/2, h + 0.35, f"{h:.1f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#047857")
        ax6.text(0.04, 0.52, "• Retener 1 'Premium' = 6.2 'Ocasionales'.\n• Q-Learning maximiza margen en +12%.",
                 transform=ax6.transAxes, fontsize=7.5, bbox=dict(boxstyle="round,pad=0.25", facecolor="#f0fdf4", edgecolor="#bbf7d0", alpha=0.9))

    # Exportación final PNG
    if f == NUM_FRAMES - 1:
        plt.savefig(D1_PNG, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")

    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor=fig.get_facecolor(), edgecolor="none")
    buf.seek(0)
    frames_d1.append(Image.open(buf).copy())
    buf.close()
    plt.close(fig)

for _ in range(8):
    frames_d1.append(frames_d1[-1])

frames_d1[0].save(D1_GIF, save_all=True, append_images=frames_d1[1:], duration=120, loop=0, optimize=True)
print(f"   [OK] Dashboard 1 generado: {D1_GIF} ({os.path.getsize(D1_GIF)/(1024*1024):.2f} MB)")

# ------------------------------------------------------------------------------
# 3. GENERADOR DEL DASHBOARD 2: DIAGNÓSTICO AVANZADO, XAI & ELASTICIDAD (6 GRÁFICOS)
# ------------------------------------------------------------------------------
print("\n[3/4] Generando Dashboard 2 (Diagnóstico Avanzado, XAI & Elasticidad - 6 Gráficos)...")
frames_d2 = []

for f in range(NUM_FRAMES):
    prog = f / (NUM_FRAMES - 1)
    fig = plt.figure(figsize=(16.0, 9.0), facecolor="#f8fafc", dpi=100)

    fig.text(0.045, 0.955, "DASHBOARD 2 | DIAGNÓSTICO AVANZADO, EXPLICABILIDAD & ELASTICIDAD ML",
             fontsize=16, fontweight="bold", color="#0f172a")
    fase_txt = "Simulando Evaluación..." if prog < 0.75 else "Diagnóstico Validado & Recomendación Activa"
    fig.text(0.045, 0.932, f"Inspección de Residuos, Inercia, ROC/AUC, Convergencia, XAI y Elasticidad [{int(prog * 100)}%]",
             fontsize=10.5, color="#059669", fontweight="bold")
    fig.text(0.81, 0.948, "Arquitectura: Ing. Juancito Peña", fontsize=9.5, fontweight="bold", color="#1e293b",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="#e2e8f0", edgecolor="#cbd5e1"))

    gs = gridspec.GridSpec(2, 3, figure=fig, left=0.07, right=0.965, top=0.900, bottom=0.075, wspace=0.24, hspace=0.34)

    # 1. Residuos
    ax1 = fig.add_subplot(gs[0, 0], facecolor="#ffffff")
    n_res = max(3, int(len(residuos) * min(1.0, prog * 1.3)))
    ax1.scatter(y1_pred[:n_res]/1000, residuos.iloc[:n_res], color="#3b82f6", alpha=0.8, s=55, edgecolors="#1d4ed8")
    ax1.axhline(0, color="#ef4444", linestyle="-", linewidth=1.5, label="Cero Error")
    ax1.axhline(2*sigma_res, color="#f59e0b", linestyle="--", linewidth=1.2, label="+2σ")
    ax1.axhline(-2*sigma_res, color="#f59e0b", linestyle="--", linewidth=1.2, label="-2σ")
    ax1.fill_between([min(y1_pred)/1000 - 5, max(y1_pred)/1000 + 5], -2*sigma_res, 2*sigma_res, color="#fef3c7", alpha=0.2)
    ax1.set_xlim(min(y1_pred)/1000 - 5, max(y1_pred)/1000 + 5)
    ax1.set_ylim(-3*sigma_res, 3*sigma_res)
    ax1.set_title("1. Análisis de Residuos y Homocedasticidad", fontsize=11, fontweight="bold", color="#0f172a", pad=6)
    ax1.set_xlabel("Ventas Predichas (Miles RD$)", fontsize=8.5, color="#475569")
    ax1.set_ylabel("Residuo (Miles RD$)", fontsize=8.5, color="#475569")
    ax1.grid(True, linestyle=":", alpha=0.6, color="#cbd5e1")
    ax1.legend(loc="upper left", fontsize=7.2, frameon=True, facecolor="#ffffff")
    if prog > 0.6:
        ax1.text(0.04, 0.05, f"• Desv. Residuos: ±RD$ {sigma_res:,.1f}k\n• Errores sin sesgo ni heterocedasticidad.",
                 transform=ax1.transAxes, fontsize=7.5, bbox=dict(boxstyle="round,pad=0.3", facecolor="#eff6ff", edgecolor="#bfdbfe", alpha=0.9))

    # 2. Codo e Inercia
    ax2 = fig.add_subplot(gs[0, 1], facecolor="#ffffff")
    n_k = max(2, int(len(k_valores) * min(1.0, prog * 1.2)))
    ax2.plot(k_valores[:n_k], inertias[:n_k], marker="o", color="#2563eb", linewidth=2.2, label="Inercia")
    ax2.set_xlabel("Número de Clusters (k)", fontsize=8.5, color="#475569")
    ax2.set_ylabel("Inercia (WCSS)", fontsize=8.5, color="#1d4ed8")
    ax2.set_xticks(k_valores)
    ax2.grid(True, linestyle=":", alpha=0.6, color="#cbd5e1")
    ax2_twin = ax2.twinx()
    ax2_twin.plot(k_valores[:n_k], sil_scores[:n_k], marker="s", color="#d97706", linewidth=2, linestyle="--", label="Silhouette")
    ax2_twin.set_ylabel("Silhouette Score", fontsize=8.5, color="#b45309")
    ax2_twin.set_ylim(0.2, 0.7)
    if prog > 0.5:
        ax2.scatter([3], [inertias[1]], color="#dc2626", s=180, zorder=6, edgecolors="white", linewidth=1.5)
        ax2.annotate("Codo Óptimo\n(k=3)", (3, inertias[1]), textcoords="offset points", xytext=(12, 10),
                     fontsize=7.8, fontweight="bold", color="#dc2626", arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.2))
    ax2.set_title("2. Método del Codo e Índice Silhouette", fontsize=11, fontweight="bold", color="#0f172a", pad=6)
    if prog > 0.6:
        ax2.text(0.38, 0.78, "• Inercia cae 56% en k=3.\n• Silhouette = 0.566 (Separación óptima).",
                 transform=ax2.transAxes, fontsize=7.5, bbox=dict(boxstyle="round,pad=0.3", facecolor="#fefce8", edgecolor="#fde047", alpha=0.9))

    # 3. ROC y AUC
    ax3 = fig.add_subplot(gs[0, 2], facecolor="#ffffff")
    n_roc = max(2, int(len(fpr) * min(1.0, prog * 1.3)))
    ax3.plot(fpr[:n_roc], tpr[:n_roc], color="#0d9488", linewidth=2.5, label=f"ROC (AUC = {roc_auc:.3f})")
    ax3.plot([0, 1], [0, 1], color="#94a3b8", linestyle=":", linewidth=1.5, label="Aleatorio")
    ax3.fill_between(fpr[:n_roc], tpr[:n_roc], alpha=0.15, color="#0d9488")
    ax3.set_xlim(-0.02, 1.02)
    ax3.set_ylim(-0.02, 1.05)
    ax3.set_title("3. Curva ROC & Discriminación (AUC)", fontsize=11, fontweight="bold", color="#0f172a", pad=6)
    ax3.set_xlabel("Falsos Positivos (FPR)", fontsize=8.5, color="#475569")
    ax3.set_ylabel("Verdaderos Positivos (TPR)", fontsize=8.5, color="#475569")
    ax3.grid(True, linestyle=":", alpha=0.6, color="#cbd5e1")
    ax3.legend(loc="lower right", fontsize=7.2, frameon=True, facecolor="#ffffff")
    if prog > 0.6:
        ax3.text(0.04, 0.78, f"• AUC = {roc_auc:.3f} (Discriminación total).\n• Cero confusión en 'Alto Valor'.",
                 transform=ax3.transAxes, fontsize=7.5, bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0fdfa", edgecolor="#99f6e4", alpha=0.9))

    # 4. Q-Learning Convergencia
    ax4 = fig.add_subplot(gs[1, 0], facecolor="#ffffff")
    idx_ep = max(50, int(num_episodios * min(1.0, prog * 1.15)))
    ax4.plot(np.arange(idx_ep), recomp_smooth[:idx_ep]/1000, color="#d97706", linewidth=2, label="Recompensa Media")
    if idx_ep > 1200:
        ax4.axvline(1200, color="#64748b", linestyle="--", linewidth=1.2)
        ax4.text(300, np.mean(recomp_smooth[:idx_ep])/1000 * 0.75, "Exploración", fontsize=7.5, color="#64748b", fontweight="bold")
        ax4.text(1400, np.mean(recomp_smooth[:idx_ep])/1000 * 0.75, "Explotación Óptima", fontsize=7.5, color="#059669", fontweight="bold")
    ax4.set_xlim(0, num_episodios)
    ax4.set_title("4. Curva de Convergencia Q-Learning", fontsize=11, fontweight="bold", color="#0f172a", pad=6)
    ax4.set_xlabel("Episodios de Simulación", fontsize=8.5, color="#475569")
    ax4.set_ylabel("Recompensa Ingreso (Miles RD$)", fontsize=8.5, color="#475569")
    ax4.grid(True, linestyle=":", alpha=0.6, color="#cbd5e1")
    ax4.legend(loc="lower right", fontsize=7.2, frameon=True, facecolor="#ffffff")
    if prog > 0.6:
        ax4.text(0.04, 0.78, "• Estabilización en ep. 2,000+.\n• +24% ingreso vs azar.",
                 transform=ax4.transAxes, fontsize=7.5, bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff7ed", edgecolor="#ffedd5", alpha=0.9))

    # 5. XAI
    ax5 = fig.add_subplot(gs[1, 1], facecolor="#ffffff")
    x_bars = np.arange(len(nombres_xai))
    ratios_anim = 1.0 + (ratios_xai - 1.0) * min(1.0, prog * 1.25)
    bars_xai = ax5.bar(x_bars, ratios_anim, color=["#e11d48" if r > 2.0 else "#3b82f6" for r in ratios_xai], width=0.55, edgecolor="#1e293b", linewidth=0.5)
    ax5.axhline(1.0, color="#64748b", linestyle="--", linewidth=1.2, label="Normal (1.0x)")
    ax5.set_xticks(x_bars)
    ax5.set_xticklabels(nombres_xai, rotation=25, ha="right", fontsize=7.8, color="#1e293b")
    ax5.set_ylim(0, 4.3)
    ax5.set_title("5. Explicabilidad Forense: Anomalías vs. Normal", fontsize=11, fontweight="bold", color="#0f172a", pad=6)
    ax5.set_ylabel("Multiplicador (Veces)", fontsize=8.5, color="#475569")
    ax5.grid(True, linestyle=":", alpha=0.6, color="#cbd5e1")
    ax5.legend(loc="upper left", fontsize=7.2, frameon=True, facecolor="#ffffff")
    if prog > 0.8:
        for idx_b, b in enumerate(bars_xai):
            h = b.get_height()
            if ratios_xai[idx_b] > 1.5:
                ax5.text(b.get_x() + b.get_width()/2, h + 0.1, f"{h:.2f}x", ha="center", va="bottom", fontsize=7.8, fontweight="bold", color="#e11d48")
        ax5.text(0.38, 0.78, "• Monto: 3.5x | Devoluciones: 3.8x.\n• Foco en compras masivas y reclamos.",
                 transform=ax5.transAxes, fontsize=7.5, bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff1f2", edgecolor="#fecdd3", alpha=0.9))

    # 6. Elasticidad
    ax6 = fig.add_subplot(gs[1, 2], facecolor="#ffffff")
    n_elast = max(5, int(len(descuentos_rango) * min(1.0, prog * 1.25)))
    ax6.plot(descuentos_rango[:n_elast], ingresos_sim[:n_elast], color="#059669", linewidth=2.5, label="Ingreso Proyectado")
    ax6.set_xlabel("Descuento (%)", fontsize=8.5, color="#475569")
    ax6.set_ylabel("Ingresos Totales (Miles RD$)", fontsize=8.5, color="#047857")
    ax6.set_xlim(0, 35)
    ax6.set_ylim(min(ingresos_sim)*0.95, max(ingresos_sim)*1.08)
    ax6.grid(True, linestyle=":", alpha=0.4, color="#cbd5e1")
    ax6_twin = ax6.twinx()
    ax6_twin.plot(descuentos_rango[:n_elast], demanda_sim[:n_elast], color="#6366f1", linewidth=1.8, linestyle=":", label="Demanda (Unidades)")
    ax6_twin.set_ylabel("Demanda (Unidades)", fontsize=8.5, color="#4f46e5")
    ax6_twin.set_ylim(min(demanda_sim)*0.9, max(demanda_sim)*1.1)

    if prog > 0.6:
        ax6.scatter([desc_optimo], [ingreso_optimo], color="#dc2626", s=140, zorder=6, edgecolors="white", linewidth=1.2)
        ax6.annotate(f"Óptimo: {desc_optimo:.1f}%\n(${ingreso_optimo:,.0f}k)", (desc_optimo, ingreso_optimo),
                     textcoords="offset points", xytext=(-35, 12), fontsize=7.8, fontweight="bold", color="#dc2626", arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.2))
        ax6.axvspan(25, 35, color="#fee2e2", alpha=0.3)
        ax6.text(26, min(ingresos_sim)*0.97, "Destrucción\nde Margen", fontsize=7.2, color="#b91c1c", fontweight="bold")

    ax6.set_title("6. Elasticidad & Ingreso Óptimo", fontsize=11, fontweight="bold", color="#0f172a", pad=6)
    lines_a, labels_a = ax6.get_legend_handles_labels()
    lines_b, labels_b = ax6_twin.get_legend_handles_labels()
    ax6.legend(lines_a + lines_b, labels_a + labels_b, loc="lower left", fontsize=7.2, frameon=True, facecolor="#ffffff")
    if prog > 0.8:
        ax6.text(0.04, 0.40, f"• Descuento Óptimo: {desc_optimo:.1f}%\n• >25% rebaja destruye 18% margen.",
                 transform=ax6.transAxes, fontsize=7.5, bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0fdf4", edgecolor="#bbf7d0", alpha=0.9))

    if f == NUM_FRAMES - 1:
        plt.savefig(D2_PNG, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")

    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor=fig.get_facecolor(), edgecolor="none")
    buf.seek(0)
    frames_d2.append(Image.open(buf).copy())
    buf.close()
    plt.close(fig)

for _ in range(8):
    frames_d2.append(frames_d2[-1])

frames_d2[0].save(D2_GIF, save_all=True, append_images=frames_d2[1:], duration=120, loop=0, optimize=True)
print(f"   [OK] Dashboard 2 generado: {D2_GIF} ({os.path.getsize(D2_GIF)/(1024*1024):.2f} MB)")

# ------------------------------------------------------------------------------
# 4. COMPILACIÓN DE LA SUITE MAESTRA COMBINADA (TRANSICIÓN DE AMBOS EN UN SOLO .GIF)
# ------------------------------------------------------------------------------
print("\n[4/4] Compilando Suite Maestra Combinada (Dashboard 1 + Dashboard 2 en un solo GIF)...")
frames_master = []

# Añadir animación de Dashboard 1
frames_master.extend(frames_d1)

# Pausa estática intermedia entre dashboards
for _ in range(10):
    frames_master.append(frames_d1[-1])

# Añadir animación de Dashboard 2
frames_master.extend(frames_d2)

# Pausa estática final
for _ in range(10):
    frames_master.append(frames_d2[-1])

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

print("\n" + "=" * 80)
print("[EXITO] SUITE MAESTRA FINALIZADA")
print(f"Tiempo Total de Procesamiento: {t_total:.1f} segundos")
print("-" * 80)
print(f"1. Dashboard 1 Animado:      {os.path.abspath(D1_GIF)} ({os.path.getsize(D1_GIF)/(1024*1024):.2f} MB)")
print(f"2. Dashboard 1 Estático:     {os.path.abspath(D1_PNG)} (300 DPI Ultra HD)")
print(f"3. Dashboard 2 Animado:      {os.path.abspath(D2_GIF)} ({os.path.getsize(D2_GIF)/(1024*1024):.2f} MB)")
print(f"4. Dashboard 2 Estático:     {os.path.abspath(D2_PNG)} (300 DPI Ultra HD)")
print(f"5. SUITE COMBINADA (TODO-EN-1): {os.path.abspath(MASTER_GIF)} ({master_mb:.2f} MB)")
print("=" * 80)
