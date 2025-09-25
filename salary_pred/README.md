# En cours 

# Lancement du back 

make runb

pas oublier de lancement docker desktop

# Lancement du front

make runf

# Avancement du projet Patron

- Lecture du fichier de config
- Accès via l'instance globale config et config.data.raw_path (accès par point et accès aux differents niveaux imbriqués)
- Model Manager qui charge / save / predict selon ce qu'on met dans le fichier config
- Peut chager quand on veut le model en cours d'utilisation
- Class FastAPI pour le serveur
- Route test qui compare mon modele de regression et le modele de sklearn
- schemas pour la sortie de modele manager avec les métriques de bases
- endpoint 
- frontend
    - Entrainement
    - Prediction
    - Métrique

Exemple pour test endpoint dans /docs

{
  "x": [
    [0],[1],[2],[3],[4],[5]
  ],
  "y": [
    27000, 38000, 43525.00, 60150.00, 56957.00, 65429.00
  ]
}

# Projet ML

Une application complète de Machine Learning. Exemple pour prédire les salaires basés sur l'expérience professionnelle (https://www.kaggle.com/datasets/rohankayan/years-of-experience-and-salary-dataset).

## Architecture

## Commande pour l'application

## ULRs

Interface Streamlit : http://localhost:8501

Documentation API : http://localhost:8000/docs

API FastAPI : http://localhost:8000

## Endpoints du Modèle

|Méthode | Route |Description|
| :--------------- |:---------------| :-----|
|GET  | /               | Informations sur l'API|
|GET  | /health         | Etat du modele et de l'API|
|GET  | /ml/metrics     | Dernière métriques du modèle|
|GET  | /ml/metrics/history     | Historique des métriques|
|POST | /ml/train       | Entraîner le modèle|
|POST | /ml/trainfile   | Entraîner le modèle sur un fichier|
|POST | /ml/predict     | Faire une prédiction|
|POST | /ml/evaluate    | Evalue le modele sur des données|

## TODO

Des tests