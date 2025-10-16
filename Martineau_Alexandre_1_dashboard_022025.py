import feature_eng
import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from io import BytesIO
import joblib
import json
import seaborn as sns
import numpy as np
import os

port = os.environ.get('PORT', 8501)

st.set_page_config(page_title="Dashboard de Crédit Scoring")

url = "https://my-scoring-app-546acd78d8fa.herokuapp.com/"

@st.cache_resource
def get_model():
    params = {"password": "Credit-Scoring-2025"}
    response = requests.get(f'{url}download_model', params=params)
    if response.status_code == 200:
        model_file = BytesIO(response.content)
        model = joblib.load(model_file)
        return model
    else:
        st.error(f"❌ Erreur {response.status_code} lors du téléchargement du modèle : {response.text}")
        return None

model = get_model()

def prepare_data_for_api(client_id, df):
    client_data = df[df['SK_ID_CURR'] == client_id].copy()
    client_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    client_data = client_data.map(lambda x: None if pd.isna(x) else x)
    response = requests.post(f'{url}predict_proba', json={"data": client_data.to_dict(orient="records")})
    response.raise_for_status()
    return response.json()[0]

@st.cache_data
def get_threshold():
    response = requests.get(f'{url}best_threshold').json()["best_threshold"]
    return response

threshold = get_threshold()

@st.cache_data
def get_list(df):
    client_ids = df['SK_ID_CURR'].unique()
    feature_list = df.drop(columns=["SK_ID_CURR", "Score", "Classe"]).columns.tolist()
    return client_ids, feature_list

df = pd.read_csv('mon_fichier.csv')

client_ids, feature_list = get_list(df)

def display_gauge(score, threshold):
    color_map = [(0.90, "darkgreen"), (0.75, "green"), (0.52, "lightgreen"), (0.45, "orange"), (0.30, "lightcoral"), (0.15, "red"), (0, "darkred")]
    bar_color = next(color for limit, color in color_map if score > limit)
    fig = go.Figure()
    fig.add_trace(go.Indicator(mode="gauge+number", value=score * 100, title={"text": "Score de Crédit (en %)"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": bar_color},
               "steps": [{"range": [0, threshold * 100], "color": "whitesmoke"}, {"range": [threshold * 100, 100], "color": "white"}],
               "threshold": {"line": {"color": "black", "width": 2}, "thickness": 1, "value": threshold * 100}}))
    st.plotly_chart(fig)

@st.cache_data
def compute_shap_values_global(df):
    json_data = json.dumps({"data": df.to_dict(orient="records")})
    data = requests.post(f'{url}data', headers={"Content-Type": "application/json"}, data=json_data).json().get("données à traiter", None)
    data = pd.DataFrame(data)
    process = pd.DataFrame(model.named_steps["preprocessor"].transform(data), columns=data.columns)
    explainer = shap.TreeExplainer(model.named_steps["model"])
    shap_values = explainer(process)
    shap_values_for_class = shap_values[..., 0]
    return shap_values, shap_values_for_class, process

shap_values, shap_values_for_class, process = compute_shap_values_global(df)

def compute_shap_values_updated(client_data_updated):
    json_data = json.dumps({"data": client_data_updated.to_dict(orient="records")})
    response = requests.post(f'{url}data', headers={"Content-Type": "application/json"}, data=json_data)
    response_data = response.json().get("données à traiter", None)
    if response_data is not None:
        new_data = pd.DataFrame(response_data)
        process_updated = pd.DataFrame(model.named_steps["preprocessor"].transform(new_data), columns=new_data.columns)
        explainer = shap.TreeExplainer(model.named_steps["model"])
        shap_values_updated = explainer(process_updated)
        shap_values_for_class_updated = shap_values_updated[..., 0]
        return shap_values_updated, shap_values_for_class_updated, process_updated
    else:
        return None, None, None

def feature_importance(df, client=None, mode="Columns"):
    if client is None:
        top_10_features_global = process.columns[np.argsort(np.abs(shap_values_for_class.values).mean(axis=0))[::-1][:10]].tolist()
        if mode == "Columns":
            return top_10_features_global
        elif mode == "Graphic":
            st.markdown("#### 📊 Feature Importance Globale (SHAP)")
            shap.initjs()
            fig = plt.figure(figsize=(10, 6))
            shap.plots.beeswarm(shap_values_for_class)
            st.pyplot(fig)
            st.write("📋 **Données utilisées pour SHAP :**")
            st.write(client_data[top_10_features_global])
        return
    use_updated_data = st.session_state.get("use_updated_data", False)
    shap_values_current = st.session_state.shap_values_for_class_updated if use_updated_data else shap_values_for_class
    process_current = st.session_state.process_updated if use_updated_data else process
    if use_updated_data:
        client_idx = 0
    else:
        client_idx = df[df['SK_ID_CURR'] == client].index[0]
    shap_instance = shap_values_current[client_idx]
    shap_instance.base_values = 0.5  
    shap_instance.data = process_current.iloc[client_idx]  
    top_10_features_local = process_current.columns[np.argsort(np.abs(shap_instance.values))[::-1][:10]].tolist()
    if mode == "Columns": 
        return top_10_features_local
    elif mode == "Graphic":
        st.markdown("#### 📊 Feature Importance Locale (SHAP)")
        shap.initjs()
        fig = plt.figure(figsize=(10, 6))
        shap.plots.waterfall(shap_instance)
        st.pyplot(fig)
        st.write("📋 **Données du client utilisées pour SHAP :**")
        st.write(client_data[top_10_features_local])

def display_feature_distribution(feature, client_value):
    if feature not in df.columns:
        st.error("⚠️ Feature non trouvée dans les données.")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df, x=feature, hue="Classe", bins=30, kde=True, alpha=0.5, ax=ax)
    ax.axvline(client_value, color='red', linestyle='dashed', linewidth=2, label="Valeur Client")
    st.pyplot(fig)

def display_bivariate_analysis(feature_x, feature_y, client_data):
    fig, ax = plt.subplots(figsize=(10, 7))
    norm = mcolors.TwoSlopeNorm(vmin=df["Score"].min(), vcenter=threshold, vmax=df["Score"].max())
    scatter = ax.scatter(df[feature_x], df[feature_y], c=df["Score"], cmap="seismic_r", norm=norm, alpha=0.75, edgecolors=None)
    client_x = client_data[feature_x].values[0]
    client_y = client_data[feature_y].values[0]
    ax.scatter(client_x, client_y, color='red', s=200, edgecolors='black', label=f"Client n°{client_id}", marker="X")
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Score de solvabilité")
    ax.set_title(f"Analyse Bi-Variée : {feature_x} vs {feature_y}")
    ax.set_xlabel(feature_x)
    ax.set_ylabel(feature_y)
    ax.legend()
    st.pyplot(fig)

if __name__ == "__main__":
    st.title("📊 Dashboard de Crédit Scoring")
    st.markdown("## 📊 Analyse du Client")
    
    if "use_updated_data" not in st.session_state:
        st.session_state.use_updated_data = False
    if "updated_score" not in st.session_state:
        st.session_state.updated_score = None
    if "modified_data" not in st.session_state:
        st.session_state.modified_data = None
    if "shap_values_for_class_updated" not in st.session_state:
        st.session_state.shap_values_for_class_updated = None
    if "shap_values_updated" not in st.session_state:
        st.session_state.shap_values_updated = None
    if "process_updated" not in st.session_state:
        st.session_state.process_updated = None
    
    if client_ids.size > 0:
        client_id = st.selectbox("🔎 Choisissez l'ID du client :", options=client_ids)
        if "last_selected_client" not in st.session_state or st.session_state.last_selected_client != client_id:
            st.session_state.last_selected_client = client_id
            st.session_state.use_updated_data = False
            st.session_state.updated_score = None
            st.session_state.modified_data = None
        client_data = df[df['SK_ID_CURR'] == client_id][feature_list]
        original_score = df[df['SK_ID_CURR'] == client_id]["Score"].values[0]
        top_10_features_global = feature_importance(df, client=None, mode="Columns")
        top_10_features_local = feature_importance(df, client=client_id, mode="Columns")
        merged_features = sorted(list(set(top_10_features_global + top_10_features_local)))
        modified_client_data = client_data.copy()
        st.sidebar.markdown("## 🛠️ Modification des informations client")
        with st.sidebar.form(key='client_form'):
            for feature in merged_features:
                feature_type = df[feature].dtype
                if feature_type == 'object':  
                    modified_client_data[feature] = st.selectbox(f"📝 Modifier {feature}", options=df[feature].unique(), index=df[feature].tolist().index(modified_client_data[feature].values[0]))
                elif feature_type in ['float64', 'int64']:
                    modified_client_data[feature] = st.number_input(f"📝 Modifier {feature}", value=float(modified_client_data[feature].values[0]))
                elif feature_type == 'bool':
                    modified_client_data[feature] = st.checkbox(f"📝 Modifier {feature}", value=bool(modified_client_data[feature].values[0]))
            submit_button = st.form_submit_button(label="Mettre à jour les informations client")
        if submit_button:
            st.session_state.modified_data = modified_client_data
            st.sidebar.success("Les informations du client ont été mises à jour avec succès !")
            st.sidebar.write("📋 **Nouvelles données du client sélectionné :**")
            st.sidebar.dataframe(modified_client_data)
        basic_button = st.sidebar.button("🔄 Voir les analyses avec les données de base")
        update_button = st.sidebar.button("🔄 Voir les analyses avec les données mises à jour")
        if basic_button:
            st.session_state.use_updated_data = False
            st.session_state.updated_score = original_score
        if update_button and st.session_state.modified_data is not None:
            st.session_state.use_updated_data = True
            modified_client_data = st.session_state.modified_data.copy()
            modified_client_data.replace([np.inf, -np.inf], np.nan, inplace=True)
            modified_client_data = modified_client_data.map(lambda x: None if pd.isna(x) else x)
            try:
                response = requests.post(f'{url}predict_proba', json={"data": modified_client_data.to_dict(orient="records")})
                response.raise_for_status()
                new_score = response.json()[0]
                st.session_state.updated_score = new_score
                shap_values_updated, shap_values_for_class_updated, process_updated = compute_shap_values_updated(modified_client_data)
                st.session_state.shap_values_updated = shap_values_updated
                st.session_state.shap_values_for_class_updated = shap_values_for_class_updated
                st.session_state.process_updated = process_updated
            except Exception as e:
                st.sidebar.error(f"Erreur API : {e}")
        client_data_to_display = st.session_state.modified_data if st.session_state.use_updated_data else client_data
        score_to_display = st.session_state.updated_score if st.session_state.updated_score is not None else original_score
        with st.expander("🔍 Analyser le client"):
            if client_id in client_ids:
                if score_to_display is not None:
                    risk_levels = [(0.90, "très faible"), (0.75, "faible"), (threshold, "légèrement faible"),(0.45, "moyen"), (0.30, "légèrement élevé"), (0.15, "élevé"), (0, "très élevé")]
                    risk_message = next(msg for limit, msg in risk_levels if score_to_display > limit)
                    st.success(f"Le client présente un risque {risk_message} de défaut de paiement.")
                    display_gauge(score_to_display, threshold)
                    st.write("📋 **Données du client sélectionné :**")
                    st.dataframe(client_data_to_display)
                else:
                    st.error("Aucune prédiction trouvée pour ce client.")
            else:
                st.warning("Aucun client trouvé avec cet ID.")
        st.markdown("## 📊 Analyse des Features")
        st.markdown("### 📊 Analyse Globale & Locale des Features")
        with st.expander("📈 Afficher l'importance globale des features"):
            feature_importance(df, client=None, mode="Graphic")
        with st.expander("📈 Afficher l'importance locale des features"):
            feature_importance(df, client=client_id, mode="Graphic")
        st.markdown("### 📊 Analyses Croisées")
        selected_feature = st.selectbox("🔎 Choisissez une première feature :", options=merged_features)
        selected_feature_2 = st.selectbox("🔎 Choisissez une deuxième feature :", options=merged_features)
        with st.expander("📊 Afficher la distribution des features"):
            if selected_feature:
                display_feature_distribution(selected_feature, client_data_to_display[selected_feature].values[0])
            if selected_feature_2:
                display_feature_distribution(selected_feature_2, client_data_to_display[selected_feature_2].values[0])
            st.write(client_data_to_display[merged_features])
        with st.expander("📊 Afficher l'analyse bi-variée"):
            display_bivariate_analysis(selected_feature, selected_feature_2, client_data_to_display)
            st.write(client_data_to_display[merged_features])
    else:
        st.warning("Aucun ID de client disponible. Veuillez vérifier les fichiers sources.")