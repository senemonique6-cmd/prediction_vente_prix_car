from pathlib import Path

import joblib as jb
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline

# ==========================================
# CONFIGURATION
# ==========================================
st.set_page_config(page_title="Prédiction du prix de vente d'une voiture", page_icon="🚗", layout="centered")

BASE_DIR = Path(__file__).parent
PIPELINE_PATH = BASE_DIR / "car_price_pipeline.joblib"  # RobustScaler + StandardScaler + Random Forest

# Ordre des variables : identique à celui utilisé à l'entraînement
FEATURES = ["Kilometrage", "Prix_Actuel", "Type_Carburant", "Type_Vendeur", "Transmission", "Age"]
CAT_COLS = ["Type_Carburant", "Type_Vendeur", "Transmission"]

# Classes des LabelEncoder (ordre alphabétique => codes 0, 1, 2 ...)
CATEGORIES = {
    "Type_Carburant": ["CNG", "Diesel", "Petrol"],
    "Type_Vendeur": ["Dealer", "Individual"],
    "Transmission": ["Automatic", "Manual"],
}

# Colonnes du CSV d'origine (anglais) -> colonnes du notebook (français)
RENAME_EN_FR = {
    "Kms_Driven": "Kilometrage",
    "Present_Price": "Prix_Actuel",
    "Fuel_Type": "Type_Carburant",
    "Seller_Type": "Type_Vendeur",
    "Age": "Age",
}


# ==========================================
# CHARGEMENT DU PIPELINE COMPLET
# ==========================================
@st.cache_resource
def load_pipeline():
    return jb.load(PIPELINE_PATH)


try:
    pipe = load_pipeline()
except FileNotFoundError as e:
    st.error(f"Fichier introuvable : {Path(e.filename).name}. Placez-le à côté de app.py.")
    st.stop()

if not isinstance(pipe, Pipeline):
    st.error("Le fichier chargé n'est pas un Pipeline (scaler + modèle). Regénérez-le depuis le notebook.")
    st.stop()


# ==========================================
# FONCTIONS DE PRÉDICTION
# ==========================================
def encode_col(series: pd.Series, col: str) -> pd.Series:
    """Accepte le texte (Petrol, Dealer, Manual...) ou directement les codes (0, 1, 2)."""
    mapping = {v.lower(): i for i, v in enumerate(CATEGORIES[col])}
    mapping.update({str(i): i for i in range(len(CATEGORIES[col]))})
    encoded = series.astype(str).str.strip().str.lower().map(mapping)
    # Les codes numériques écrits en float (ex. "1.0")
    if encoded.isna().any():
        num = pd.to_numeric(series, errors="coerce")
        encoded = encoded.fillna(num)
    if encoded.isna().any() or not encoded.isin(range(len(CATEGORIES[col]))).all():
        raise ValueError(f"Valeurs invalides dans « {col} ». Valeurs acceptées : {CATEGORIES[col]}")
    return encoded.astype(int)


def predict(df_features: pd.DataFrame) -> np.ndarray:
    x = df_features[FEATURES].to_numpy(dtype=float)
    return pipe.predict(x)


# ==========================================
# INTERFACE
# ==========================================
st.title("🚗 Prédiction du prix de vente d'une voiture")
st.write("Ce modèle (Random Forest) prédit le prix de vente d'une voiture à partir de ses caractéristiques.")

tab1, tab2 = st.tabs(["Prédiction simple", "Prédiction multiple (CSV)"])

# ---------- Onglet 1 : prédiction simple ----------
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        kilometrage = st.number_input("Kilométrage", min_value=0, value=30000, step=500)
        prix_actuel = st.number_input("Prix actuel", min_value=0.0, value=6.0, step=0.1)
        age = st.number_input("Âge de la voiture (années)", min_value=0, value=5, step=1)
    with col2:
        carburant = st.selectbox("Type de carburant", CATEGORIES["Type_Carburant"], index=2)
        vendeur = st.selectbox("Type de vendeur", CATEGORIES["Type_Vendeur"])
        transmission = st.selectbox("Transmission", CATEGORIES["Transmission"], index=1)

    if st.button("Prédire le prix de vente", type="primary"):
        row = pd.DataFrame(
            [{
                "Kilometrage": kilometrage,
                "Prix_Actuel": prix_actuel,
                "Type_Carburant": CATEGORIES["Type_Carburant"].index(carburant),
                "Type_Vendeur": CATEGORIES["Type_Vendeur"].index(vendeur),
                "Transmission": CATEGORIES["Transmission"].index(transmission),
                "Age": age,
            }]
        )
        y_pred = float(predict(row)[0])
        st.success(f"Prix de vente prédit : **{y_pred:.2f}**")

# ---------- Onglet 2 : prédiction multiple ----------
with tab2:
    st.write(
        "Importez un fichier CSV contenant les colonnes : "
        "`Kilometrage`, `Prix_Actuel`, `Type_Carburant`, `Type_Vendeur`, `Transmission`, `Age`."
    )
    uploaded = st.file_uploader("Importer un fichier CSV", type=["csv"])

    if uploaded is not None:
        try:
            data = pd.read_csv(uploaded).rename(columns=RENAME_EN_FR)
            missing = [c for c in FEATURES if c not in data.columns]
            if missing:
                st.error(f"Colonnes manquantes : {missing}")
            else:
                X = data[FEATURES].copy()
                for c in CAT_COLS:
                    X[c] = encode_col(X[c], c)
                data["Prix_Vente_Predit"] = np.round(predict(X), 2)

                st.dataframe(data, use_container_width=True)
                st.download_button(
                    "Télécharger les prédictions",
                    data=data.to_csv(index=False).encode("utf-8"),
                    file_name="predictions.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")
