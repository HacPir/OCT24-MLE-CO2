# CO2 DATASCIENTEST Project
### Prédiction des émissions de CO2 d'un Véhicule Léger - Déploiement du Project
#### Pour DataScientest - Soutenance de projet - Parcours DevOps


Ce projet vise à prédire les émissions de CO₂ (WLTP) des véhicules à partir de caractéristiques techniques telles que la masse, la cylindrée, la puissance, la consommation de carburant et le type de carburant. Plusieurs modèles de machine learning sont comparés (Random Forest, Régression Linéaire, KNN), avec et sans inclusion des informations sur les marques, et une optimisation via TPOT est réalisée.
Ce projet vise à déployer une solution de Machine Learning dans le respect des règles du cycle de vie DevOps.

<picture>
 <img align="center" alt="Cycle DevOps" src="https://browserstack.wpenginepowered.com/wp-content/uploads/2023/02/DevOps-Lifecycle.jpg">
</picture>

Ainsi nous vous présentons ce projet qui vise à automatiser la récupération d'un dataset, entraîner un modèle puis le mettre à disposition via une plateforme API. Notre solution permet également la supervision et le surveillance de toutes les phases de notre système. 

L'application finale permet la prédiction des émissions de CO₂ (WLTP) d'un véhicules à partir de caractéristiques techniques (masse, la cylindrée, la puissance, cylindrée, système de réduction des émissions, la consommation de carburant et le type de carburant). Nous proposons une étude où plusieurs modèles de Machine Learning peuvent être entraîné afin de comparer les résultats, soit les algorithmes de Forêt d'arbres décisionnels (Random Forest), Régression Linéaire et Méthode des K plus proches voisins (KNN).
[avec et sans inclusion des informations sur les marques.]

## Table des matières

- [Présentation du Projet](#présentation-du-projet)
- [Structure du Projet](#structure-du-projet)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Modèles et Données](#modèles-et-données)
- [Téléchargement du Dataset](#téléchargement-du-dataset)
- [Pré-processing et Concaténation des Datasets](#pré-processing-et-concaténation-des-datasets)
- [Architecture](#architecture)
- [Axes d'Amélioration](#axes-damélioration)
- [Licence](#licence)
- [Contributions](#contributions)

## Présentation du Projet

Face aux enjeux climatiques et aux régulations strictes sur les émissions de CO₂, il est crucial de développer des outils permettant d'estimer l'impact environnemental des véhicules. Ce projet a pour objectifs :
- de récupérer un dataset distant,
- de réaliser le pré-processing nécessaire,
- d'entraîner un modèle,
- d'évaluer le modèle,
- de déployer le nouveau modèle,
- de surveiller les métrics

## Structure du Projet

  nov24_bds_co2
├── LICENSE
├── README.md
├── __pycache__
│   └── streamlit.cpython-313.pyc
├── app.py
├── data
│   ├── DF2023-22-21_Concat_Finale_2.csv
│   ├── features_rf_opt_tpot.pkl
│   └── shap_values_knn.pkl
├── images
│   ├── Countplot_Mk.png
│   ├── Exemple_Detect_Outliers.png
│   └── voitureco2.png
├── models
│   ├── pipeline_knn_ext.pkl
│   ├── pipeline_knn_sm.pkl
│   ├── pipeline_linear_regression_ext.pkl
│   ├── pipeline_linear_regression_sm.pkl
│   ├── pipeline_random_forest_opt_tpot.pkl
│   ├── pipeline_random_forest_opt_tpot_sm.pkl
│   └── pipeline_random_forest_sm.pkl
└── src
    ├── concatenate_datasets.py
    ├── model.py
    ├── pre_processing.py
    ├── print_tree.py
    └── regression_analysis.py

## Installation

1. Cloner le dépôt :

       git clone https://dagshub.com/tiffany.dalmais/OCT24_MLOPS_CO2.git
       cd NOV24-BDS-CO2

2. Créer un environnement virtuel (optionnel, mais recommandé) :

       python -m venv venv
       source venv/bin/activate   # Sur Windows : venv\Scripts\activate

3. Installer les dépendances :

       pip install -r requirements.txt

4. Installer DVC :

       python3 -m pip install --upgrade pip # Mise à jour de la bibliothèque des paquets Python
       pip install dvc

5. Installer MLflow et dagshub :

       python3 -m pip install mlflow dagshub # Installation des deux applications


6. Configurer l'authentification (optionnel mais utile pour éviter de ressaisir ses identifiants à chaque fois) : 
       a. Ouvrir un terminal et créer/modifier le fichier ~/.netrc :

       nano ~/.netrc

    b. Ajouter les lignes suivantes au fichier : 

       machine dagshub.com # Nom du serveur distant
       login nom_utilisateur # Nom d'utilisateur
       password token_dagshub # Token d'authentification

    c. Enregistrer les modifications et quitter l'éditeur de texte : 

       Ctrl+O, Enter, Ctrl+X

    d. Sécuriser le fichier : 

       chmod 600 ~/.netrc


8. Exécuter la pipeline pour reproduire les étapes : 

       dvc repro

10. Gestion des commits et des push : 
En cas de modification de la pipeline et/ou des scripts : 
    a. Ajouter et committer les modifications (code, dvc.yaml, dvc.lock, etc.) :
    
        git add .
        git commit -m "Description du commit"

    b. Pousser le code vers le remote Git :
    
        git push origin main

    c. Pousser les données volumineuses via DVC :

        dvc push
       
## Utilisation

Pour lancer l'application Streamlit :

       streamlit run app.py

L'application s'ouvrira dans votre navigateur à l'adresse [http://localhost:8501](http://localhost:8501).

## Modèles et Données

- Les modèles entraînés (.pkl) seront enregistrés dans le dossier `models/`.
- Pour générer ces fichiers, exécutez le script `model.py` situé dans le dossier `src/`. Une fois lancé, les fichiers de modélisation seront automatiquement sauvegardés dans `models/`.
- Le prétraitement des données est réalisé via le script `pre_processing.py`.
- Les fichiers d'entrée et de sortie pour l'entraînement et l'évaluation se trouvent dans le dossier `data/`.


## Téléchargement du Dataset

Le dataset se récupère automatiquement grâce au script 'recup_raw_data.py' qui permet de lancer une requête SQL afin de récupérer les informations nécessaire à l'entraînement de notre modèle. Ces informations sont contenus dans les colonnes : Year, Mk, Cn, M (kg), Ewltp (g/km), Ft, Ec (cm3), Ep (KW), Erwltp (g/km) et Fc. 
Afin que le modèle soit toujours à jour concernant les nouveaux véhicules, nous vous conseillons d'automatiser le lancement de la Pipeline complète (commande : dvc repro) grâce à un Cron Job (commande : 0 0 1 */3 * /usr/local/bin/dvc repro). 
Toutefois vous pouvez également lancer simplement le script 'recup_raw_data.py' depuis votre terminal afin de récupérer simplement le dataset.

## Pré-processing et Concaténation des Datasets

Le pré-processing a été effectué séparément pour chaque dataset des années 2023, 2022 et 2021.  
Les fichiers pré-traités sont nommés :
- `DF2023_Processed_2.csv`
- `DF2022_Processed_2.csv`
- `DF2021_Processed_2.csv`

Pour obtenir le dataset final, ces fichiers ont été concaténés à l'aide du script `concatenate_datasets.py`.  
Exécutez le script en ligne de commande :

       python concatenate_datasets.py

Le fichier résultant, `DF2023-22-21_Concat_Finale_2.csv`, doit être placé dans le dossier `src_new/data`.

## Architecture

|     Application    | Framework |
|-------------------:|-----------|
|   Automatisation   | Cron Job  |
|     Versionning    | DVC       |
|      Repository    | DagsHub   |
|   Contenarisation  | Docker    |
| Suivi Entraînement | MLFlow    |
|     Monitoring     | Prometheus|
|  Tableau de bord   | Grafana   |
|     Interface      | Streamlit |


## Axes d'Amélioration

- Intégration de nouvelles variables explicatives.
- Optimisation des hyperparamètres avec d'autres techniques.
- Déploiement sur une plateforme cloud pour un accès public.

## Licence

Ce projet est distribué sous la licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.
