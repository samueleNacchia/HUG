import pandas as pd
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import seaborn as sns

# ==================
#   DATA CLEANING
# ==================

def getCleanedData(df_input):
    """
    Esegue SOLO l'azione di pulizia sui dati:
    - Rimozione colonne manuali
    - Rimozione Null e Duplicati
    - Rimozione Outlier Numerici (IQR 0.30 - 0.70) con Report
    - Rimozione righe con CGPA < 4
    - Rimozione Outlier Categorici (Freq < 1%) con Report
    """
    # 1. Copia e parametri locali
    df = df_input.copy()
    target = 'Depression'
    df_size = len(df)
    print("\n--- INIZIO PULIZIA ---")
    print(f"Righe iniziali: {df_size}")

    # 2. Rimozione colonne non informative
    columns_to_drop = ['id', 'Have you ever had suicidal thoughts ?']
    df = df.drop(columns=columns_to_drop, errors='ignore')

    # 3. Pulizia righe: Null e Duplicati
    df = df.dropna()
    df = df.drop_duplicates()

    # 4. Rimozione Outlier Numerici (IQR 0.30-0.70)
    print("\n--- RIMOZIONE OUTLIER NUMERICI (IQR) ---")

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if target in numeric_cols:
        numeric_cols = numeric_cols.drop(target)

    for col in numeric_cols:
        Q1 = df[col].quantile(0.30)
        Q3 = df[col].quantile(0.70)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        # Contiamo prima di rimuovere
        n_before = len(df)

        # Applichiamo il filtro
        df = df[(df[col] >= lower) & (df[col] <= upper)]

        n_removed = n_before - len(df)

        # Se abbiamo rimosso qualcosa, lo stampiamo
        if n_removed > 0:
            print(f"Feature '{col}': rimossi {n_removed} outlier.")

    # 5. Rimozione righe con CGPA < 4
    if 'CGPA' in df.columns:
        n_before = len(df)
        df = df[df['CGPA'] >= 5]
        if len(df) < n_before:
            print(f"Filtro CGPA < 5: rimossi {n_before - len(df)} studenti.")

    # 6. Rimozione Outlier Categorici (Freq < 1%)
    print("\n--- RIMOZIONE OUTLIER CATEGORICI (< 1%) ---")

    # Identifichiamo le colonne categoriche (escluso Target e City)
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    # City viene esclusa perché è normale avere tante città con poche persone
    cols_to_exclude = ['City','Degree', target]

    cols_to_clean = [c for c in cat_cols if c not in cols_to_exclude]
    THRESHOLD_PERCENT = 0.01

    for col in cols_to_clean:
        value_counts = df[col].value_counts(normalize=True)
        rare_values = value_counts[value_counts < THRESHOLD_PERCENT].index.tolist()

        if rare_values:
            n_before = len(df)
            df = df[~df[col].isin(rare_values)]
            n_removed = n_before - len(df)

            print(f"Feature '{col}': rimosse {n_removed} righe.")
            print(f"   -> Valori rari eliminati: {rare_values}")

    # 3. PULIZIA SPECIFICA PER 'DEGREE' (Solo 'Others')
    if 'Degree' in df.columns:
        n_before = len(df)
        # Rimuoviamo solo le righe dove Degree è 'Others'
        df = df[df['Degree'] != 'Others']
        n_removed = n_before - len(df)

        if n_removed > 0:
            print(f"Feature 'Degree': rimosse {n_removed} righe.")
            print(f"   -> Valore eliminato: ['Others']")

    print(f"\nRighe finali post-cleaning: {len(df)}")
    print(f"\nRighe eliminate con il data cleaning: {df_size-len(df)}")
    print("------------------------------------------")

    return df