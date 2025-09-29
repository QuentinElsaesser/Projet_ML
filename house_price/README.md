# Projet ML

Une application complète de Machine Learning basé sur le dataset https://www.kaggle.com/datasets/arunjangir245/boston-housing-dataset/data.

# Lancement du back 

make runb

pas oublier de lancement docker desktop

# Lancement du front

make runf

# Avancement du projet

- Lecture du fichier de configuration
- Accès via l'instance globale config (accès par point et accès aux différents niveaux imbriqués)
- Model Manager qui charge/sauvegarde/prédit selon la configuration
- Peut changer le modèle en cours d'utilisation à tout moment
- Classe FastAPI pour le serveur
- Schémas pour la sortie du Model Manager avec les métriques de base
- Endpoint
- Frontend
    - Entraînement
    - Prédiction
    - Métriques
- La target doit toujours s'appeler 'y' dans le dataset
- Modèles pris en charge : "ridge", "lasso", "random_forest", "xgboost", "lightgbm", "mlp"

## Exemple

### Train / Evaluate

{
  "x": [[5.1, 3.5, 1.4, 0.2], [6.2, 3.4, 5.4, 2.3]],
  "y": [0, 1],
  "model_type": "random_forest"
}

### Predict

{
  "x": [[5.1, 3.5, 1.4, 0.2], [6.2, 3.4, 5.4, 2.3]],
  "model_type": "random_forest"
}

## Architecture

## ULRs

Interface Streamlit : http://localhost:8501

Documentation API : http://localhost:8000/docs

API FastAPI : http://localhost:8000

## Endpoints du modèle

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

## Dataset

This data frame contains the following columns:

crim
per capita crime rate by town.

zn
proportion of residential land zoned for lots over 25,000 sq.ft.

indus
proportion of non-retail business acres per town.

chas
Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).

nox
nitrogen oxides concentration (parts per 10 million).

rm
average number of rooms per dwelling.

age
proportion of owner-occupied units built prior to 1940.

dis
weighted mean of distances to five Boston employment centres.

rad
index of accessibility to radial highways.

tax
full-value property-tax rate per $10,000.

ptratio
pupil-teacher ratio by town.

black
1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town.

lstat
lower status of the population (percent).

medv/y
median value of owner-occupied homes in $1000s.