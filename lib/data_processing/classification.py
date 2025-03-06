from sklearn.metrics import classification_report, confusion_matrix
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.base import accuracy_score
import pandas as pd
import numpy as np


def train_model():

  # Filtering
  features = np.array(features)
  targets = np.array(targets)

  mask = ~np.isnan(features).any(axis=1)

  # Filtrar as linhas válidas
  features = features[mask]
  targets = targets[mask]



  # Normalizing==========================
  scaler = StandardScaler()
  X_train = scaler.fit_transform(features)
  y_train = targets


  # Definir os hiperparâmetros para busca
  param_grid = {
      'C': [0.01, 0.1, 1, 10, 100],  # Regularização
      'penalty': ['l1', 'l2'],  # Penalização (L1 para LASSO, L2 para Ridge)
      'solver': ['liblinear']  # Necessário para suportar L1 e L2
  }

  # Configurar o GridSearchCV com validação cruzada de 5 folds
  grid_search = GridSearchCV(LogisticRegression(max_iter=5000), param_grid, n_jobs=-1)

  # Treinar o modelo
  grid_search.fit(X_train, y_train)

  # Melhor combinação de hiperparâmetros
  print("Melhores hiperparâmetros:", grid_search.best_params_)


  #============================================================================
  # TEST 
  #============================================================================
  features = []
  targets = []

  # Filtering
  features = np.array(features)
  targets = np.array(targets)

  mask = ~np.isnan(features).any(axis=1)

  # Filtrar as linhas válidas
  features = features[mask]
  targets = targets[mask]

  # Normalizing==========================
  scaler = StandardScaler()
  X_test = scaler.fit_transform(features)
  y_test = targets

  best_model = grid_search.best_estimator_
  y_pred = best_model.predict(X_test)


  # weights and bias
  print(best_model.coef_)
  print(best_model.intercept_)

  # Avaliar o modelo
  accuracy = accuracy_score(y_test, y_pred)
  conf_matrix = confusion_matrix(y_test, y_pred)
  class_report = classification_report(y_test, y_pred)

  print("Acurácia:", accuracy)
  print("Matriz de Confusão:\n", conf_matrix)
  print("Relatório de Classificação:\n", class_report)
