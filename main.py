import streamlit as st
import numpy as np
import pandas as pd
import json

st.title("Comparison between methods")

st.subheader('Parameters: ')
anormal = st.slider("Anomaly digit: ", min_value=0, max_value=9, value=0, step=1)
threshold = st.slider("Threshold: ", min_value=0., max_value=0.2, step=0.01, value=0.05)

with open(f"models_infos/VAE_RecLoss/p_values_{anormal}.json", "r") as file:
    dic_p_values_test_vae = json.load(file)

with open(f"models_infos/DeepOneClass/ONE/p_values_{anormal}.json", "r") as file:
    dic_p_values_test_one = json.load(file)

with open(f"models_infos/DeepOneClass/SOFT/p_values_{anormal}.json", "r") as file:
    dic_p_values_test_soft = json.load(file)

results = []
for digit in range(10):

    p_values_test_vae, len_test_vae = dic_p_values_test_vae[str(digit)]
    p_values_test_vae = np.asarray(p_values_test_vae)

    n_rejets_vae = (p_values_test_vae < threshold).sum().item()
    percentage_rejected_vae = n_rejets_vae / len_test_vae



    p_values_test_one, len_test_one = dic_p_values_test_one[str(digit)]
    p_values_test_one = np.asarray(p_values_test_one)

    n_rejets_one = (p_values_test_one < threshold).sum().item()
    percentage_rejected_one = n_rejets_one / len_test_one



    p_values_test_soft, len_test_soft = dic_p_values_test_soft[str(digit)]
    p_values_test_soft = np.asarray(p_values_test_soft)

    n_rejets_soft = (p_values_test_soft < threshold).sum().item()
    percentage_rejected_soft = n_rejets_soft / len_test_soft

    # Ajouter les données au tableau
    results.append({

        "Digit": digit,
        "Anormal": "Yes" if digit == anormal else "No",

        "Rejections VAE": f"{n_rejets_vae}/{len_test_vae}",
        "Rejections ONE": f"{n_rejets_one}/{len_test_one}",
        "Rejections SOFT": f"{n_rejets_soft}/{len_test_soft}",


        "Rejection Rate VAE": f"{percentage_rejected_vae:.3%}",
        "Rejection Rate ONE": f"{percentage_rejected_one:.3%}",
        "Rejection Rate SOFT": f"{percentage_rejected_soft:.3%}"
    })

df_results = pd.DataFrame(results)

st.subheader('Table results: ')
st.table(df_results)