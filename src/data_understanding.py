import pandas as pd
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import seaborn as sns
import os

if __name__ == "__main__":

    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parent.parent
    DATASET_PATH = BASE_DIR / "student_depression.csv"

    df = pd.read_csv(DATASET_PATH)

    target = 'Depression'

    # --- ANALISI ---
    print("\n=== DATA UNDERSTANDING ===")
    print(f"Dimensioni iniziali del dataset: {df.shape}")
    print(f"Totale valori nulli rilevati: {df.isnull().sum().sum()}")
    print(f"Numero di righe duplicate: {df.duplicated().sum()}")

    # Controllo finale valori negativi
    num_cols = df.select_dtypes(include=['number']).columns
    neg_cols = [c for c in num_cols if (df[c] < 0).any()]

    if neg_cols:
        print(f"\nColonne con negativi: {neg_cols}")
    else:
        print("\nNessun valore negativo trovato.")

    # --- ANALISI FEATURE NUMERICHE ---
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    if target in numerical_cols: numerical_cols.remove(target)

    numerical_corr = {}
    numerical_anomaly = {}

    for col in numerical_cols:
        col_data = df[col].dropna()
        numerical_corr[col] = col_data.corr(df[target])

        # Statistiche distribuzione
        skewness_val = skew(col_data)
        kurt_val = kurtosis(col_data)
        std_val = col_data.std()

        # Flag anomalie (tua logica originale)
        numerical_anomaly[col] = abs(skewness_val) > 1 or kurt_val > 5 or std_val < 0.01

    # --- ANALISI FEATURE CATEGORICHE ---
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    categorical_dependence = {}
    categorical_anomaly = {}

    for col in categorical_cols:
        col_data = df[col].dropna()
        # Calcolo dipendenza tramite crosstab
        ct = pd.crosstab(col_data, df[target], normalize='index')
        max_diff = abs(ct[1] - ct[0]).max()
        categorical_dependence[col] = max_diff

        # Flag distribuzione (tua logica originale)
        n_unique = col_data.nunique()
        value_counts = col_data.value_counts(normalize=True) * 100
        categorical_anomaly[col] = n_unique <= 1 or value_counts.max() > 90

    # --- STAMPE RISULTATI ANALISI ---
    sorted_corr = dict(sorted(numerical_corr.items(), key=lambda x: abs(x[1]), reverse=True))
    print("\nFeature numeriche più correlate alla target:")
    for col, corr in sorted_corr.items():
        print(f"{col}: {corr:.3f}")

    sorted_dep = dict(sorted(categorical_dependence.items(), key=lambda x: x[1], reverse=True))
    print("\nFeature categoriche con maggiore dipendenza dalla target:")
    for col, dep in sorted_dep.items():
        print(f"{col}: {dep:.3f}")


    print("\n=== CONTEGGIO OCCORRENZE (Esclusi id, Age, CGPA e City) ===")

    # Lista delle colonne da escludere
    exclude_cols = ['id', 'Age','CGPA', 'City']

    for col in df.columns:
        # Controlla se la colonna corrente NON è tra quelle da escludere
        if col not in exclude_cols:
            print(f"\n--- Valori per la feature: {col} ---")
            print(df[col].value_counts()) # Stampa solo i conteggi assoluti

    # --- IDENTIFICAZIONE CANDIDATI ALLA RIMOZIONE ---
    numerical_candidates = [col for col in numerical_cols if numerical_anomaly[col] or abs(numerical_corr[col]) < 0.05]
    print("\nFeature numeriche anomale candidate per rimozione:", numerical_candidates)

    categorical_candidates = [col for col in categorical_cols if categorical_anomaly[col] or categorical_dependence[col] < 0.05]
    print("Feature categoriche anomale candidate per rimozione:", categorical_candidates)

    if 'Have you ever had suicidal thoughts ?' in df.columns:
        ct = pd.crosstab(
        df['Have you ever had suicidal thoughts ?'],
        df['Depression'],
        normalize='index'
        )
        print("\n", ct)