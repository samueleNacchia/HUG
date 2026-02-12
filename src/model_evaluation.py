import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score
import data_preparation
from models import SmartDepressionClassifier

# ==============================================================================
# 1. SETUP E CARICAMENTO
# ==============================================================================
X = data_preparation.X
y = data_preparation.y

model = joblib.load('modello_finale.pkl')

cat_features = ['Gender', 'Sleep Duration', 'Dietary Habits', 'Family History of Mental Illness', 'Degree_level', 'Age_group']
num_features = ['Academic Pressure', 'Study Satisfaction', 'Work/Study Hours', 'Financial Stress', 'CGPA_30']

# ==============================================================================
# 2. PREPARAZIONE DATI (Binning Automatico)
# ==============================================================================
X_eval = X.copy()

for col in num_features:
    if col == 'CGPA_30':
        # Per il CGPA usiamo i quartili (Basso, Medio, Alto)
        try:
            X_eval[f'{col}_slice'] = pd.qcut(X_eval[col], q=3, labels=['Basso', 'Medio', 'Alto'], duplicates='drop')
        except ValueError:
            X_eval[f'{col}_slice'] = X_eval[col].astype(str)
    else:
        X_eval[f'{col}_slice'] = X_eval[col].astype(str)

# ==============================================================================
# 3. ANALISI MASSIVA
# ==============================================================================
all_slices = []
features_to_analyze = cat_features + [f'{c}_slice' for c in num_features]

for feat in features_to_analyze:
    for val in X_eval[feat].unique():
        mask = X_eval[feat] == val
        if mask.sum() < 30: continue # Ignoriamo fette troppo piccole (statisticamente irrilevanti)

        y_slice = y[mask]
        y_pred = model.predict(X_eval[mask].drop(columns=[f'{c}_slice' for c in num_features if f'{c}_slice' in X_eval.columns]))

        all_slices.append({
            'Feature': feat.replace('_slice', ''),
            'Valore': val,
            'F1-Score': f1_score(y_slice, y_pred, zero_division=0),
            'Recall': recall_score(y_slice, y_pred, zero_division=0),
            'Supporto': len(y_slice)
        })

df_report = pd.DataFrame(all_slices).sort_values('F1-Score')
global_f1 = f1_score(y, model.predict(X))

# =============
# 4. OUTPUT
# =============

# --- TOP 5 PEGGIORI ---
print("\nLE 5 FETTE CON PERFORMANCE PEGGIORI:")
print(df_report.head(8).to_string(index=False))
df_worst = df_report.head(8).copy()
df_worst['Label'] = df_worst['Feature'] + ": " + df_worst['Valore'].astype(str)

# --- TOP 5 MIGLIORI ---
print("\nLE 5 FETTE CON PERFORMANCE MIGLIORI:")
print(df_report.tail(5).sort_values('F1-Score', ascending=False).to_string(index=False))
df_best = df_report.tail(5).sort_values('F1-Score', ascending=False).copy()
df_best['Label'] = df_best['Feature'] + ": " + df_best['Valore'].astype(str)

# ======================
# 4. VISUALIZZAZIONE
# ======================

plt.figure(figsize=(10, 6))
sns.set_style("white")

# 1. Definizione della palette
worst_palette = sns.cubehelix_palette(n_colors=len(df_worst), start=2.8, rot=.1, light=.85, dark=.35, reverse=True)

ax1 = sns.barplot(
    data=df_worst, x='F1-Score', y='Label', hue='Label',
    palette=worst_palette, legend=False, edgecolor=".2",
    zorder=2
)

linea_media = plt.axvline(
    x=global_f1, color='#4169E1', linestyle='--',
    linewidth=2, alpha=0.8, label=f'Media Globale ({global_f1:.2f})',
    zorder=3
)

# 2. Personalizzazione assi e titolo
plt.title('Slices con performance critiche\n',
          fontsize=14, fontweight='bold', pad=20, color='#4B0082')
plt.xlim(0, 1.0)

# 3. Etichette dei valori sulle barre
for container in ax1.containers:
    ax1.bar_label(container, fmt='%.3f', padding=5, fontweight='bold')

plt.legend(handles=[linea_media], loc='upper left', bbox_to_anchor=(1, 1), frameon=False)

plt.tight_layout()
plt.show()

# ==============================================================================
# 3. GRAFICO 2: I PUNTI DI FORZA (MIGLIORI)
# ==============================================================================
plt.figure(figsize=(10, 6))
sns.set_style("white")

# 1. Definizione della palette
best_palette = sns.cubehelix_palette(n_colors=len(df_best), start=2.8, rot=.1, light=.85, dark=.35, reverse=True)

ax1 = sns.barplot(
    data=df_best, x='F1-Score', y='Label', hue='Label',
    palette=best_palette, legend=False, edgecolor=".2",
    zorder=2
)

linea_media = plt.axvline(
    x=global_f1, color='#4169E1', linestyle='--',
    linewidth=2, alpha=0.8, label=f'Media Globale ({global_f1:.2f})',
    zorder=3
)

# 2. Personalizzazione assi e titolo
plt.title('Slices con performance migliori\n',
          fontsize=14, fontweight='bold', pad=20, color='#4B0082')
plt.xlim(0, 1.0)

# 3. Etichette dei valori sulle barre
for container in ax1.containers:
    ax1.bar_label(container, fmt='%.3f', padding=5, fontweight='bold')

plt.legend(handles=[linea_media], loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
plt.tight_layout()
plt.show()