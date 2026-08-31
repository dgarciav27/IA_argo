# Application de l'IA au contrôle qualité des données Argo

Ce dépôt contient le code développé dans le cadre d'un stage portant sur l'application de méthodes d'apprentissage automatique (Machine Learning) et d'apprentissage profond (Deep Learning) au contrôle qualité automatique des profils de flotteurs Argo (données océanographiques : température, salinité, pression), sur trois bassins océaniques (Atlantique, Indien, Pacifique) et sur la période 2018–2022.

L’objectif est d’identifier automatiquement les profils anormaux  afin d'assister les experts scientifiques dans la chaîne de contrôle qualité.
## Sommaire

- [Contexte](#contexte)
- [Modèles implémentés](#modèles-implémentés)
- [Résultats principaux](#résultats-principaux)
- [Structure du dépôt](#structure-du-dépôt)
- [Installation](#installation)
- [Reproduire les résultats](#reproduire-les-résultats)
- [Suivi des expériences (MLflow)](#suivi-des-expériences-mlflow)
- [Benchmark de vitesse d'entraînement](#benchmark-de-vitesse-dentraînement)
- [Auteur](#auteur)

## Contexte

Le réseau Argo déploie des flotteurs autonomes qui mesurent en continu la température, la salinité et la pression des océans. Chaque profil doit passer un contrôle qualité (QC) avant diffusion. Ce projet explore si des modèles d'IA peuvent détecter automatiquement les profils anormaux (`is_bad = 1`) à partir de variables dérivées des profils verticaux, en complément du contrôle qualité expert existant.

Le problème est traité comme une **classification binaire déséquilibrée** (profils valides vs. anormaux), avec un split **temporel** (train/val/test) pour éviter toute fuite d'information (data leakage) entre périodes.

## Modèles implémentés

| Catégorie | Modèles |
|---|---|
| Machine Learning supervisé | Random Forest, XGBoost, LightGBM |
| Machine Learning non supervisé | Isolation Forest |
| Deep Learning | CNN-1D, Transformer, Autoencodeur + LightGBM (hybride) |

Chaque modèle est entraîné **indépendamment par bassin océanique** (Atlantique / Indien / Pacifique), avec recherche d'hyperparamètres (`HalvingRandomSearchCV` pour les modèles à base d'arbres), calibration du seuil de décision sur le jeu de validation, et évaluation finale sur le jeu de test.

## Résultats principaux

Évaluation sur le jeu de test, avec une moyenne calculée sur les trois bassins océaniques (Atlantique, Indien et Pacifique).
| Modèle | F1-score | ROC-AUC | Accuracy |
|---|---:|---:|---:|
| **LightGBM** | **83.2 %** | **90.1 %** | **84.3 %** |
| XGBoost | 83.1 % | 90.1 % | 84.2 % |
| Random Forest | 82.8 % | 89.9 % | 83.8 % |
| CNN-1D | 73.2 % | 81.1 % | 74.7 % |
| Autoencoder + LightGBM | 68.5 % | 76.4 % | 70.3 % |
| Transformer | 66.6 % | 71.7 % | 67.9 % |
| Isolation Forest | 56.7 % | 58.6 % | 60.9 % |

**LightGBM est retenu comme modèle de référence** : performances quasi-équivalentes à XGBoost et Random Forest.

Les modèles à base d'arbres de décision (ML supervisé) surpassent nettement les approches de Deep Learning testées et l'approche non supervisée, tout en affichant une performance stable entre eux.

## Structure du dépôt

```
IA_argo/
├── 0.Dowload_data/
│   └── argo_download_process.ipynb     # Téléchargement des fichiers NetCDF depuis le GDAC
├── 1.Preprocess_ML/
│   └── preprocessing.ipynb             # Prétraitement pour les modèles ML classiques
├── 2.Training_ML/
│   ├── IF/                             # Isolation Forest
│   ├── LightGBM/
│   ├── RF/                             # Random Forest
│   └── XGBoost/
├── 3.Preprocess_DL/
│   └── 1.preprocessed_fixed.py         # Prétraitement pour les modèles Deep Learning
├── 4.Training_DL/
│   ├── Autoencoder/
│   ├── CNN/
│   └── Transformer/
├── Others/
├── Plots/
│   ├── autoencoder_plots.ipynb
│   ├── comparison_dl_ml_multiocean.ipynb
│   └── dataset_plots_2018.ipynb
├── requirements.txt                    # Dépendances Python
└── README.md
```

> ⚠️ Les données brutes/prétraitées (`.nc`, `.parquet`) et les artefacts de modèles entraînés (`.pkl`, MLflow) sont volumineux et **ne sont pas versionnés** sur GitHub (voir `.gitignore`). Voir la section [Installation](#installation) pour savoir où placer vos propres données.

## Installation

```bash
git clone https://github.com/dgarciav27/IA_argo.git
cd IA_argo

# Créer l'environnement (conda recommandé)
conda create -n argo_env python=3.10
conda activate argo_env

pip install -r requirements.txt
```

### Dépendances principales

```
pandas
numpy
seaborn
scikit-learn
xgboost
xarray
lightgbm
torch          # pour CNN-1D / Transformer / Autoencodeur
mlflow
joblib
pyarrow        # lecture/écriture parquet
scipy
cartopy
xgboost
netcdf4
```

### Données requises

Ce dépôt ne contient pas les données Argo brutes. Pour reproduire les résultats :

1. Télécharger les fichiers NetCDF Argo (profils 2018–2022) depuis le GDAC en exécutant `0.Dowload_data/argo_download_process.ipynb`.
2. Lancer le prétraitement adapté (ML ou DL, voir ci-dessous) pour générer les splits `train/val/test` utilisés par les scripts d'entraînement.

## Reproduire les résultats

### 1. Prétraitement (obligatoire avant tout entraînement)

Deux pipelines de prétraitement distincts sont disponibles selon la famille de modèles ciblée :

- **`1.Preprocess_ML/preprocessing.ipynb`** — pour les modèles ML classiques (Random Forest, XGBoost, LightGBM, Isolation Forest).
- **`3.Preprocess_DL/1.preprocessed_fixed.py`** — pour les modèles Deep Learning (CNN-1D, Transformer, Autoencodeur).

> **Note :** le prétraitement est configurable et permet de choisir : le **type de split** (temporel ou par plateforme/flotteur), le sens de la trajectoire (cycles ascendants uniquement ou non), et l'application ou non d'un **sous-échantillonnage (undersampling)** de la classe majoritaire pour atténuer le déséquilibre des classes. Se référer aux paramètres définis en tête du notebook/script pour ajuster ces options selon le cas d'usage souhaité.

Ceci génère, pour chaque bassin, les splits `train`/`val`/`test` ainsi que la liste des variables (`feature_cols`) utilisés par les scripts d'entraînement.

### 2. Entraînement d'un modèle

Chaque sous-dossier de `2.Training_ML/` (`IF`, `LightGBM`, `RF`, `XGBoost`) et `4.Training_DL/` (`Autoencoder`, `CNN`, `Transformer`) contient le script/notebook d'entraînement correspondant. Chacun suit la même logique : baseline → recherche d'hyperparamètres (CV sur train, pour les modèles ML) → calibration du seuil (sur val) → évaluation finale (sur test) → sauvegarde des artefacts (modèle, métriques, rapports).

Les trois bassins (Atlantique, Indien, Pacifique) sont généralement entraînés séquentiellement dans une même exécution (voir le dictionnaire `OCEANS` en tête de chaque script — adapter les chemins d'entrée/sortie à votre installation).

### 3. Résultats générés

Pour chaque modèle et chaque bassin, les scripts sauvegardent typiquement :
- le modèle entraîné (`.pkl`)
- un rapport de classification 
- les résultats de la recherche d'hyperparamètres (pour les modèles ML)
- les prédictions sur le jeu de test
- les données nécessaires à la génération de graphiques, utilisées ensuite dans les notebooks du dossier `Plots/`

### 4. Visualisation et comparaison des modèles

Le notebook **`Plots/comparison_dl_ml_multiocean.ipynb`** centralise la comparaison des performances de tous les modèles (ML et DL) sur les trois bassins océaniques. `Plots/dataset_plots_2018.ipynb` et `Plots/autoencoder_plots.ipynb` fournissent respectivement l'exploration des données brutes et l'analyse spécifique du modèle hybride Autoencodeur + LightGBM.

## Suivi des expériences (MLflow)

Les entraînements des modèles sont journalisés automatiquement dans MLflow (paramètres, métriques, modèle). Pour visualiser les expériences :

```bash
mlflow ui --backend-store-uri sqlite:///<chemin_vers_votre_mlflow.db>
```

Puis ouvrir [http://localhost:5000](http://localhost:5000). 

## Auteur

D. R. Garcia Valencia — Stage "Application de l'IA au contrôle qualité Argo"
[github.com/dgarciav27](https://github.com/dgarciav27)
