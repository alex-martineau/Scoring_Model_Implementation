from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import traceback

app = Flask(__name__)

# Charger le modèle
try:
    model = joblib.load("mlruns_reduced/model.pkl")
except Exception as e:
    print(f"Erreur lors du chargement du modèle: {e}")
    model = None

try:
    original_columns = joblib.load("mlruns_reduced/columns.pkl")
except Exception as e:
    print(f"Erreur lors du chargement des colonnes : {e}")
    original_columns = None

try:
    with open("mlruns_reduced/best_threshold", "r") as f:
        best_threshold = float(f.read().strip())
except Exception as e:
    print(f"Erreur lors du chargement du seuil : {e}")
    best_threshold = None

# Vérifiez que les fichiers critiques sont bien chargés
if model is None or original_columns is None or best_threshold is None:
    raise RuntimeError("Les fichiers nécessaires n'ont pas été chargés correctement.")

# Route par défaut
@app.route("/", methods=["GET"])
def home():
    return (
        "Bienvenue sur le serveur de prédiction !\n\n\n"
        "Pour consulter le seuil optimal de prédiction, dirigez-vous vers : '/predict'\n\n"
        "Pour consulter les features, dirigez-vous vers : '/features'\n\n"
        "Pour consulter l'ensemble des données que vous nous avez envoyées, dirigez-vous vers : '/data'\n\n\n"
        "En vous souhaitant une bonne expérimentation :)"
    )
@app.route("/features", methods=["GET"])
def get_features():
    if original_columns is None:
        return jsonify({"error": "Les colonnes ne sont pas disponibles"}), 500
    return jsonify({"columns": original_columns})

@app.route("/best_threshold", methods=["GET"])
def get_threshold():
    if best_threshold is None:
        return jsonify({"error": "Le seuil optimal n'est pas disponible"}), 500
    return jsonify({"best_threshold": best_threshold})

# Fonction de prédiction
def predict_with_threshold(X):
    y_pred_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_pred_prob > float(best_threshold)).astype(int)
    return y_pred

# Fonction de prétraitement
def preprocess_dataset(df):
    df = df.copy()  # Ne pas modifier l'objet original
    missing_cols = set(original_columns) - set(df.columns)
    for col in missing_cols:
        df[col] = 0
    df = df[original_columns]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.astype('float64')
    return df

@app.route("/data", methods=["POST"])
def posted():
    try:
        data = request.get_json()
        return jsonify({"données à traiter": data})
    except Exception as e:
        return jsonify({"error": str(e), "stack_trace": traceback.format_exc()}), 500

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame(data["data"])
        prepared_data = preprocess_dataset(df)
        prediction = predict_with_threshold(prepared_data)
        predictor = {}
        for i in range(len(prediction.tolist())):
            predictor[f"client n°{df['SK_ID_CURR'].tolist()[i]}"] = "Yes" if prediction.tolist()[i] == 0 else "No"
        return jsonify({"prediction": predictor})
    except Exception as e:
        return jsonify({"error": str(e), "stack_trace": traceback.format_exc()}), 500

@app.route("/predict_proba", methods=["POST"])
def predict_proba():
    try:
        data = request.get_json()
        df = pd.DataFrame(data["data"])
        prepared_data = preprocess_dataset(df)
        y_pred_proba = model.predict_proba(prepared_data)
        proba_response = {}
        for i, proba in enumerate(y_pred_proba):
            proba_response[f"client n°{df['SK_ID_CURR'].iloc[i]}"] = {"niveau de solvabilité": float(proba[0]), "seuil": float(best_threshold)}
        return jsonify({"prediction": proba_response})
    except Exception as e:
        return jsonify({"error": str(e), "stack_trace": traceback.format_exc()}), 500

if __name__ == '__main__':
    app.run(debug=False)