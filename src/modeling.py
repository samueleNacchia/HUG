import pandas as pd
import sys
import warnings
import joblib
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# ==============================================================================
# 1. CARICAMENTO DATI
# ==============================================================================
print("⏳ Caricamento pipeline...")

datasets = {}

try:
    import pipeline  # Deve esistere pipeline.py
    datasets['Pipeline'] = (pipeline.X, pipeline.y, pipeline.preprocessor)
except ImportError:
    print("❌ pipeline.py non trovato")
    sys.exit()

# ==============================================================================
# 2. SETUP CV
# ==============================================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
leaderboard_data = []

# ==============================================================================
# 3. CICLO MODELLI
# ==============================================================================
for pipe_name, (X, y, preprocessor) in datasets.items():

    print("\n" + "=" * 60)
    print(f"📂 DATASET: {pipe_name}")
    print("=" * 60)

    # --------------------------------------------------------------------------
    # A. RANDOM FOREST (Sostituito a Decision Tree)
    # --------------------------------------------------------------------------
    algo_name = "Random Forest"
    print(f"\n🔹 Modello: {algo_name}")

    pipe_rf = Pipeline(steps=[
        ('preprocess', preprocessor),
        ('model', RandomForestClassifier(
            class_weight='balanced',
            random_state=42
        ))
    ])

    print(" ...GridSearch in corso")
    # Parametri ottimizzati per Random Forest
    param_grid = {
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

    grid = GridSearchCV(
        pipe_rf,
        param_grid,
        cv=cv,
        scoring=scoring,
        refit='accuracy',
        n_jobs=-1
    )

    grid.fit(X, y)
    best_rf = grid.best_estimator_
    print(f"\n✅ Migliore combinazione per {algo_name}")
    print(grid.best_params_)

    # Salvataggio risultati in leaderboard
    leaderboard_data.append({
        'Source': pipe_name,
        'Algorithm': algo_name,
        'Accuracy': grid.cv_results_['mean_test_accuracy'][grid.best_index_],
        'Precision': grid.cv_results_['mean_test_precision'][grid.best_index_],
        'Recall': grid.cv_results_['mean_test_recall'][grid.best_index_],
        'F1': grid.cv_results_['mean_test_f1'][grid.best_index_],
        'ROC AUC': grid.cv_results_['mean_test_roc_auc'][grid.best_index_]
    })

    # Feature Importance RF
    print(f"\n⭐ Feature Importance ({algo_name})")
    rf_model = best_rf.named_steps['model']
    feature_names = best_rf.named_steps['preprocess'].get_feature_names_out()
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': rf_model.feature_importances_}).sort_values(by='Importance', ascending=False)
    print(importance_df.head(10))

    # --------------------------------------------------------------------------
    # B. LOGISTIC REGRESSION
    # --------------------------------------------------------------------------
    algo_name = "Logistic Regression"
    print(f"\n🔹 Modello: {algo_name}")

    pipe_log = Pipeline(steps=[
        ('preprocess', preprocessor),
        ('model', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
    ])

    scores_log = cross_validate(pipe_log, X, y, cv=cv, scoring=scoring, n_jobs=-1)

    leaderboard_data.append({
        'Source': pipe_name,
        'Algorithm': algo_name,
        'Accuracy': scores_log['test_accuracy'].mean(),
        'Precision': scores_log['test_precision'].mean(),
        'Recall': scores_log['test_recall'].mean(),
        'F1': scores_log['test_f1'].mean(),
        'ROC AUC': scores_log['test_roc_auc'].mean()
    })




# ==============================================================================
# 4. LEADERBOARD E GRAFICI
# ==============================================================================
df_results = pd.DataFrame(leaderboard_data).sort_values(by='Accuracy', ascending=False)

print("\n🏆 CLASSIFICA FINALE")
print(df_results.to_string(index=False, float_format="%.4f"))


# ==============================================================================
# 4. LEADERBOARD FINALE E SALVATAGGIO
# ==============================================================================
df_results = pd.DataFrame(leaderboard_data).sort_values(by='Accuracy', ascending=False)
print("\n🏆 CLASSIFICA FINALE")
print(df_results.to_string(index=False, float_format="%.4f"))

print("\n💾 Salvataggio del modello finale...")

# CORREZIONE: Usiamo l'unico dataset disponibile caricato sopra
X, y, preprocessor = datasets['Pipeline']

# Salviamo il modello che ha performato meglio (esempio: la Logistic Regression finale)
pipe_final = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('model', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
])
pipe_final.fit(X, y)

filename = 'modello_finale.pkl'
joblib.dump(pipe_final, filename)
print(f"✅ Modello salvato come {filename}")