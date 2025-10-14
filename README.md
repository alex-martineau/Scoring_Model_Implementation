# Objectif du Projet

Ce projet consiste à créer une API de prédiction du score de solvabilité d'un client en utilisant un modèle de machine learning. Cette API a été déployée sur **Heroku** et permet de réaliser des prédictions pour déterminer si un client est solvable ou non. Le modèle a été préalablement entraîné et enregistré, avec un suivi complet des expérimentations via **MLFlow**.


# Structure du Projet

Voici l'organisation des fichiers dans le dépôt :

├── Procfile # Fichier de configuration pour Heroku
├── app.py # Code de l'API
├── deploy.yml # Pipeline CI/CD pour déploiement sur Heroku
├── mlruns_reduced/ # Dossier contenant les fichiers du modèle
│ ├── best_threshold # Le seuil optimal de prédiction
│ ├── columns.pkl # Liste des colonnes du dataset utilisé
│ └── model.pkl # Modèle de machine learning entraîné
├── requirements.txt # Liste des dépendances Python
├── runtime.txt # Version de Python pour Heroku
├── test_app.py # Fichier de tests unitaires pour l'API
└── README.md # Documentation du projet

# Fonctionnalités de l'API

L'API expose plusieurs points de terminaison pour interagir avec le modèle de prédiction.

## Route principale (/)
- Affiche un message de bienvenue et décrit les routes disponibles.

## Obtenir les caractéristiques du modèle (/features)
- Méthode : GET
- Retourne la liste des caractéristiques utilisées pour entraîner le modèle.

## Obtenir le seuil optimal de prédiction (/best_threshold)
- Méthode : GET
- Retourne le seuil optimal utilisé pour classer un client comme solvable ou non.

## Prédiction (/predict)
- Méthode : POST
- Permet de soumettre des données au modèle pour obtenir une prédiction (Yes ou No).

Exemple de requête POST :

{
  "data": [
    {
      "SK_ID_CURR": 155312,
      "CODE_GENDER": 0,
      "FLAG_OWN_CAR": 0,
      "FLAG_OWN_REALTY": 0,
      ...
    }
  ]
}

## Prédiction avec Probabilités (/predict_proba)
- Méthode : POST
- Permet de soumettre des données au modèle pour obtenir une prédiction accompagnée de sa probabilité.

Exemple de réponse :

{
  "prediction": {
    "client n°155312": {
      "niveau de solvabilité": 0.23,
      "seuil": 0.48
    }
  }
}

## Soumettre des données pour traitement (/data)
- Méthode : POST
- Permet d'envoyer des données brutes à l'API pour les afficher, mais pas pour les traiter directement.

# Déploiement sur Heroku

L'API est configurée pour être déployée sur Heroku. Voici les étapes pour déployer l'API sur votre propre instance Heroku : 
- Connectez-vous à Heroku CLI : heroku login
- Créez une application Heroku : heroku create my-scoring-app
- Déployez l'application sur Heroku : git push heroku main
- Accédez à votre API déployée : https://my-scoring-app-546acd78d8fa.herokuapp.com

## Déploiement Continu avec GitHub Actions

Le fichier deploy.yml est configuré pour déployer automatiquement l'API sur Heroku à chaque push sur la branche main via un pipeline CI/CD.

# Tests Unitaires

Des tests unitaires sont inclus pour vérifier le bon fonctionnement de l'API. Les tests peuvent être exécutés en utilisant pytest :
- Installez pytest : pip install pytest
- Exécutez les tests : pytest
Les tests se trouvent dans le fichier test_app.py.

# MLFlow et Suivi des Expérimentations

Le suivi des expérimentations est effectué à l'aide de MLFlow, permettant de suivre les performances du modèle au fur et à mesure des ajustements. Vous pouvez accéder à l'interface web de MLFlow pour visualiser les expériences, les paramètres et les métriques associées.

Pour exécuter MLFlow localement, suivez les étapes suivantes :
- Installez MLFlow : pip install mlflow
- Lancez l'interface web de MLFlow : mlflow ui
L'interface sera accessible à l'adresse http://localhost:5000.

Pour toute question ou demande supplémentaire, n'hésitez pas à me contacter !