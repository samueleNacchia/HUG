import joblib
import pandas as pd

# Carica modello
model = joblib.load("modello_finale.pkl")
print("✅ Modello caricato correttamente.\n")

# Trova il ColumnTransformer
ct_name = None
for step_name in model.named_steps:
    if 'transform' in str(model.named_steps[step_name]).lower():
        ct_name = step_name
        break
ct = model.named_steps[ct_name]
print(f"✅ ColumnTransformer trovato: '{ct_name}'\n")

# Definisci le colonne numeriche che ti interessano
numeric_cols = ['Age', 'Academic Pressure', 'Study Satisfaction',
                'Work/Study Hours', 'CGPA_30']

# -----------------------------
# CATEGORICHE
# -----------------------------
print("Categorie attese dal modello per ogni colonna categorica:\n")
for name, transformer, cols in ct.transformers_:
    if hasattr(transformer, 'categories_'):
        for col, cats in zip(cols, transformer.categories_):
            print(f"{col}: {list(cats)}")
print("\n")

# -----------------------------
# NUMERICHE
# -----------------------------
print("Valori numerici attesi (range statistico):\n")
for col in numeric_cols:
    # Se vuoi, puoi anche usare info statistiche dal training, qui simulo range tipico
    print(f"{col}: tipo=float, valori reali da controllare con input")


