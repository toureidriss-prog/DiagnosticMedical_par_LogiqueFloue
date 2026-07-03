import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from fuzzy_model import predict, explain_rule, rules_df

st.set_page_config(page_title="Diagnostic Flou", layout="wide")

st.title("🩺 Diagnostic Cardiaque basé sur Logique Floue")

menu = st.sidebar.selectbox("Menu", ["Dashboard", "Diagnostic", "Explicabilité"])

# ===================== DASHBOARD =====================
if menu == "Dashboard":

    st.header("Performance du système")

    st.metric("Nombre de règles", len(rules_df))

    st.write("Aperçu des règles")
    st.dataframe(rules_df)

# ===================== DIAGNOSTIC =====================
elif menu == "Diagnostic":

    st.header("Tester un patient")

    thalach = st.slider("Thalach", 60, 220, 150)
    oldpeak = st.slider("Oldpeak", 0.0, 7.0, 1.5)
    ca = st.selectbox("CA", [0,1,2,3])
    thal = st.selectbox("Thal", [3,6,7])

    if st.button("Diagnostiquer"):

        cls, val = predict(thalach, oldpeak, ca, thal)

        # aucune règle activée
        if cls is None:

            st.warning(
                "⚠️ Aucune règle de la base de connaissances "
                "n'a été activée pour cette combinaison d'entrée."
            )

        else:

            labels = {
                0:"Sain",
                1:"Faible",
                2:"Modéré",
                3:"Sévère",
                4:"Très sévère"
            }

            st.success(f"Classe : {labels[cls]}")
            st.info(f"Score flou : {val:.2f}")

# ===================== EXPLICABILITÉ =====================
elif menu == "Explicabilité":

    st.header("Règle activée")

    thalach = st.slider("Thalach", 60, 220, 150)
    oldpeak = st.slider("Oldpeak", 0.0, 7.0, 1.5)
    ca = st.selectbox("CA", [0,1,2,3])
    thal = st.selectbox("Thal", [3,6,7])

    rule, act = explain_rule(thalach, oldpeak, ca, thal)

if rule is None:

    st.warning(
        "⚠️ Aucune règle n'a été activée. "
        "Le système ne peut pas fournir d'explication."
    )

else:

    st.code(rule)
    st.write("Activation :", act)