# pipeline2.py
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns
# =================================================================
# CONFIGURAZIONE
# =================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "student_depression.csv"

TARGET = 'Depression'

# =================================================================
# FASE 1: CARICAMENTO & CLEANING
# =================================================================
print("--- PIPELINE 2 | DATA PREPARATION ---")
print("--- FASE 1: CARICAMENTO & CLEANING ---")

df_raw = pd.read_csv(DATASET_PATH)

from data_cleaning import getCleanedData
print("Richiesta dataset pulito...")
df = getCleanedData(df_raw)

print(f"Dataset ricevuto! Dimensioni: {df.shape}")
print(df.head())

# =================================================================
# FASE 2: FEATURE ENGINEERING – DEGREE
# =================================================================
print("\n--- FASE 2: TRASFORMAZIONE DEGREE ---")

def map_degree(deg):
    if deg == "'Class 12'":
        return 'Diploma'
    elif isinstance(deg, str) and deg.startswith('B'):
        return 'Titolo_primo_livello'
    elif isinstance(deg, str) and (deg.startswith('M') or deg in ['PhD', 'MD']):
        return 'Titolo_secondo_livello'
    else:
        return 'Titolo_secondo_livello'

df['Degree_level'] = df['Degree'].apply(map_degree)

print("Distribuzione Degree_level:")
print(df['Degree_level'].value_counts())

# =================================================================
# FASE 3: FEATURE ENGINEERING – AGE
# =================================================================

print("\n--- FASE 3: TRASFORMAZIONE AGE ---")

def map_age_custom(age):
    if age < 22:
        return '18-21'
    elif age < 26:
        return '22-25'
    elif age < 30:
        return '26-29'
    else:
        return '30+'

# Applicazione della trasformazione
df['Age_group'] = df['Age'].apply(map_age_custom)
print("Statistiche Age_group:")
print(df['Age_group'].describe())

# =================================================================
# FASE 4: FEATURE ENGINEERING – CGPA
# =================================================================
print("\n--- FASE 4: TRASFORMAZIONE CGPA ---")

df['CGPA_30'] = 2.4 * df['CGPA'] + 6

print("Statistiche CGPA_30:")
print(df['CGPA_30'].describe())

# =================================================================
# FASE 5: FEATURE SELECTION
# =================================================================
print("\n--- FASE 5: FEATURE SELECTION ---")

columns_to_drop = [
    'City',
    'Work Pressure',
    'Job Satisfaction',
    'Profession',
    'Degree',
    'CGPA',
    'Age'
]

df = df.drop(columns=columns_to_drop, errors='ignore')
print(f"Feature rimosse: {columns_to_drop}")
print(f"Dimensioni dopo Feature Selection: {df.shape}")

# =================================================================
# FIX: Financial Stress come numerica
# =================================================================
df['Financial Stress'] = pd.to_numeric(df['Financial Stress'], errors='coerce')

# =================================================================
# FASE 6: SPLIT X / y (RAW)
# =================================================================
print("\n--- FASE 6: Split X / y (RAW) ---")

X = df.drop(columns=[TARGET])
y = df[TARGET]
print("Split effettuato.")

# =================================================================
# FASE 7: DEFINIZIONE PREPROCESSOR (NON FITTATO)
# =================================================================
print("\n--- FASE 7: DEFINIZIONE PREPROCESSING ---")

categorical_features = X.select_dtypes(include=['object']).columns.tolist()
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print("Feature categoriche:", categorical_features)
print("Feature numeriche:", numeric_features)

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(
            drop='first',
            handle_unknown='ignore',
            sparse_output=False
        ), categorical_features),
        ('num', StandardScaler(), numeric_features)
    ],
    remainder='drop'
)

# =================================================================
# EXPORT
# =================================================================
__all__ = ['X', 'y', 'preprocessor']
