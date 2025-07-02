# Tweet Project - Classification de Tweets Catastrophes

## Description

Ce projet a pour but de classifier des tweets en deux catégories :  
- **Catastrophe** (ex : incendie, inondation, etc.)  
- **Normal** (tweets classiques sans événement catastrophique)

---

## Installation

### Prérequis

- Python 3.8+  
- `pip` installé  
- (Optionnel) Docker pour containeriser l'application

### Étapes

1. Cloner le dépôt :
   ```bash
   git clone <URL_DU_DEPOT>
   cd tweet_project

2. Créer un environnement virtuel recommandé : 
   ``` bash
   python -m venv tweet_env
    source tweet_env/bin/activate   # Linux/MacOS
    tweet_env\Scripts\activate      # Windows

3. Installer les dépendances : 
   ``` bash
   pip install -r requirements.txt

4. Lancer les pyttest en local ou  le script principal avec le modèle pré-entraîné : 
   ``` bash
   pytest tests/test_preprocessing.py 

   pytest tests/test_modeling.py 

   python main.py


5. Utilisation avec Docker : 

    À la racine du projet (où se trouve le Dockerfile), lancer :
   ``` bash
   docker build --no-cache -t tweet_project .

#### Autres commandes utiles: 
   ``` bash
docker run --rm tweet_project pytest
```
Pour l' Exécution des tests (
 pytest 
)

   ``` bash
docker run --rm tweet_project
```
Pour l'exécution du script principal (main.py)