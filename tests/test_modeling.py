# tests/test_modeling.py

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from src.modeling import build_pipeline, train_and_evaluate
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

@pytest.fixture
def toy_df():
    # Création d'un mini jeu pour tests
    return pd.DataFrame({
        'clean_text': [
            "earthquake damage",  # catastrophe
            "happy birthday",     # non catastrophe
            "flood in city",      # catastrophe
            "love this day"       # non catastrophe
        ],
        'target': [1, 0, 1, 0]
    })

def test_build_pipeline_default():
    pipe = build_pipeline()
    # Doit être un Pipeline sklearn
    assert hasattr(pipe, 'fit') and hasattr(pipe, 'predict')
    # Contient un vecteur TF-IDF et un classifieur
    names = [name for name, _ in pipe.steps]
    assert 'tfidf' in names
    assert 'clf' in names

def test_pipeline_train_predict(toy_df: pd.DataFrame):
    pipe = build_pipeline()
    # S'entraîne sans erreur
    pipe.fit(toy_df['clean_text'], toy_df['target'])
    preds = pipe.predict(toy_df['clean_text'])
    # Forme des prédictions
    assert isinstance(preds, (list, np.ndarray))
    assert len(preds) == len(toy_df)

def test_pipeline_with_svm(toy_df: pd.DataFrame):
    # Test avec un classifieur SVM dans le pipeline
    pipe = build_pipeline(model=SVC())
    pipe.fit(toy_df['clean_text'], toy_df['target'])
    preds = pipe.predict(toy_df['clean_text'])
    assert len(preds) == len(toy_df)

def test_train_and_evaluate_structure(toy_df: pd.DataFrame):
    result = train_and_evaluate(toy_df, test_size=0.5, random_state=0)
    # Vérifier que toutes les clés sont présentes
    for key in ['pipeline', 'X_test', 'y_test', 'preds', 'scores']:
        assert key in result
    # Les métriques doivent être dans scores
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'report']:
        assert metric in result['scores']

def test_metrics_values(toy_df: pd.DataFrame):
    out = train_and_evaluate(toy_df, test_size=0.5, random_state=1)
    # Scores entre 0 et 1
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        val = out['scores'][metric]
        assert 0.0 <= val <= 1.0

def test_empty_text_behavior():
    # Comportement sur texte vide ou très court
    pipe = build_pipeline()
    pipe.fit(["some text", "another example"], [0, 1])
    preds = pipe.predict(["", "   ", "a", "ok"])
    # Doit renvoyer 0 ou 1, sans erreur
    assert all(p in [0, 1] for p in preds)

def test_train_and_evaluate_with_cv(toy_df: pd.DataFrame):
    # Test de la validation croisée dans train_and_evaluate
    res = train_and_evaluate(toy_df, test_size=0.5, random_state=42, cv=2)
    assert 'cv_scores' in res
    cv_scores = res['cv_scores']
    assert 'mean_f1' in cv_scores and 'std_f1' in cv_scores
    assert isinstance(cv_scores['mean_f1'], float)
    assert isinstance(cv_scores['std_f1'], float)

def test_train_and_evaluate_with_different_vectorizers(toy_df: pd.DataFrame):
    # Test avec CountVectorizer
    res_count = train_and_evaluate(toy_df, use_count=True, test_size=0.5)
    assert 'scores' in res_count
    # Test avec TF-IDF (par défaut)
    res_tfidf = train_and_evaluate(toy_df, use_count=False, test_size=0.5)
    assert 'scores' in res_tfidf
