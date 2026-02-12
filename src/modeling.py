import pandas as pd
import sys
import warnings
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# ==============================================================================
# 1. CARICAMENTO DATI
# ==============================================================================
print("⏳ Caricamento dati da data_preparation...")

try:
    import data_preparation
    # Estraiamo direttamente i componenti necessari
    X = data_preparation.X
    y = data_preparation.y
    preprocessor = data_preparation.preprocessor
    print(f"✅ Dati caricati correttamente. Dimensioni: {X.shape}")
except ImportError:
    print("❌ data_preparation.py non trovato")
    sys.exit()

# ==============================================================================
# 2. SETUP CV E LEADERBOARD
# ==============================================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
leaderboard_data = []

# ==============================================================================
# 3. ADDESTRAMENTO MODELLI
# ==============================================================================

# --------------------------------------------------------------------------
# A. RANDOM FOREST
# --------------------------------------------------------------------------
algo_name_rf = "Random Forest"
print("\n" + "=" * 60)
print(f"🔹 Modello: {algo_name_rf}")
print("=" * 60)

pipe_rf = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('model', RandomForestClassifier(class_weight='balanced', random_state=42))
])

print(" ...GridSearch in corso")
param_grid_rf = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [None, 10, 20],
    'model__min_samples_leaf': [5, 10]
}

scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc'
}

grid_rf = GridSearchCV(
    pipe_rf,
    param_grid_rf,
    cv=cv,
    scoring=scoring,
    refit='accuracy',
    n_jobs=-1
)

grid_rf.fit(X, y)
best_rf = grid_rf.best_estimator_

print(f"\n✅ Migliore combinazione per {algo_name_rf}:")
print(grid_rf.best_params_)

# Salvataggio risultati RF
leaderboard_data.append({
    'Algorithm': algo_name_rf,
    'Accuracy': grid_rf.cv_results_['mean_test_accuracy'][grid_rf.best_index_],
    'Precision': grid_rf.cv_results_['mean_test_precision'][grid_rf.best_index_],
    'Recall': grid_rf.cv_results_['mean_test_recall'][grid_rf.best_index_],
    'F1': grid_rf.cv_results_['mean_test_f1'][grid_rf.best_index_],
    'ROC AUC': grid_rf.cv_results_['mean_test_roc_auc'][grid_rf.best_index_]
})

# Feature Importance
print(f"\n⭐ Feature Importance ({algo_name_rf})")
rf_model = best_rf.named_steps['model']
feature_names = best_rf.named_steps['preprocess'].get_feature_names_out()
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)
print(importance_df.head(10))


# --------------------------------------------------------------------------
# B. LOGISTIC REGRESSION
# --------------------------------------------------------------------------
algo_name_log = "Logistic Regression"
print("\n" + "=" * 60)
print(f"🔹 Modello: {algo_name_log}")
print("=" * 60)

pipe_log = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('model', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
])

print(" ...Cross-Validazione in corso")
scores_log = cross_validate(
    pipe_log,
    X, y,
    cv=cv,
    scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
    n_jobs=-1
)

# Stampe risultati Logit
for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
    mean_score = scores_log[f'test_{metric}'].mean()
    std_score  = scores_log[f'test_{metric}'].std()
    print(f" {metric:10s}: {mean_score:.3f} ± {std_score:.3f}")

leaderboard_data.append({
    'Algorithm': algo_name_log,
    'Accuracy': scores_log['test_accuracy'].mean(),
    'Precision': scores_log['test_precision'].mean(),
    'Recall': scores_log['test_recall'].mean(),
    'F1': scores_log['test_f1'].mean(),
    'ROC AUC': scores_log['test_roc_auc'].mean()
})

# Coefficienti Beta
print(f"\n📈 Coefficienti Beta ({algo_name_log})")
pipe_log.fit(X, y)
model_log = pipe_log.named_steps['model']
coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Beta': model_log.coef_[0],
    'Odds_Ratio': np.exp(model_log.coef_[0])
}).sort_values(by='Beta', key=abs, ascending=False)
print(coef_df.head(10))

# ==============================================================================
# 4. LEADERBOARD FINALE E SALVATAGGIO
# ==============================================================================
df_results = pd.DataFrame(leaderboard_data).sort_values(by='Accuracy', ascending=False)

print("\n" + "=" * 80)
print("🏆 CLASSIFICA FINALE")
print("=" * 80)
print(df_results.to_string(index=False, float_format="%.4f"))

# ==============================================================================
# 5. GENERAZIONE GRAFICI
# ==============================================================================

# --- GRAFICO A BARRE COMPARATIVO ---
print("\n📊 Generazione grafico a barre...")

# Trasformiamo il dataframe in formato "long" per seaborn
df_plot = df_results.melt(id_vars=['Algorithm'],
                          value_vars=['Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC'],
                          var_name='Metrica', value_name='Punteggio')

plt.figure(figsize=(12, 7))
sns.set_style("whitegrid")

# Creazione del barplot (palette simile a quella dell'immagine)
ax = sns.barplot(data=df_plot, x='Metrica', y='Punteggio', hue='Algorithm', palette='Purples')

# Personalizzazione estetica
plt.title('Confronto Performance: Random Forest vs Logistic Regression', fontsize=14, fontweight='bold')
plt.ylabel('Punteggio (Score)')
plt.xlabel('Metriche di Valutazione')
plt.ylim(0.70, 1.0)  # Zoom per evidenziare le differenze come nell'immagine
plt.legend(title='Configurazione', loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=2)

# Aggiunta delle etichette numeriche sopra le barre
for container in ax.containers:
    ax.bar_label(container, fmt='%.3f', padding=3, fontsize=9)

plt.tight_layout()
plt.show()


# --- MATRICE DI CONFUSIONE AGGREGATA (Esempio per Logistic Regression) ---
print("\n📊 Generazione Matrice di Confusione...")

# Per la matrice aggregata dobbiamo rifare una rapida predizione
# (o usare i risultati della cross-validation se salvati)
# Qui usiamo il modello Logistic Regression già fittato precedentemente
y_pred = pipe_log.predict(X)
cm = confusion_matrix(y, y_pred)

fig, ax = plt.subplots(figsize=(8, 6))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=['No Depression', 'Depression']
)

# Plot con lo stile blu dell'immagine
disp.plot(cmap='Blues', ax=ax, values_format='d')

plt.title('Aggregated Confusion Matrix', fontsize=13)
plt.suptitle(f'Modello: {algo_name_log}', fontsize=10)

plt.show()

# Salvataggio del modello vincitore (il primo della classifica)
best_model_name = df_results.iloc[0]['Algorithm']
print(f"\n💾 Salvataggio del modello vincitore ({best_model_name})...")

if best_model_name == algo_name_rf:
    final_model = best_rf
else:
    final_model = pipe_log # pipe_log è già stata fittata per i coefficienti beta

filename = 'modello_finale.pkl'
joblib.dump(final_model, filename)
print(f"✅ Modello salvato come {filename}")