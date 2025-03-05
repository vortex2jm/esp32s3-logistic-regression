from sklearn.metrics import classification_report, confusion_matrix
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.base import accuracy_score
import pandas as pd
import numpy as np


def train_model():
  window_time = 10
  window_sample_size = window_time * 400   # sample rate: 400Hz
  gap = 200

  features = []
  targets = []

  for person in range(1, 6):
    for session in range(1, 6):
      df_step = pd.read_csv(f"data/model_data/g_force/train/{person}_step{session}_acc.csv")
      df_squat = pd.read_csv(f"data/model_data/g_force/train/{person}_squat{session}_acc.csv")

      for i in range(0, len(df_step) - window_sample_size + 1, gap):
        window = df_step.iloc[i : i + window_sample_size]  # sliding window

        magnitude = window["magnitude"]
        X = window["acc_x"]
        Y = window["acc_y"]
        Z = window["acc_z"]

        features.append([
          np.mean(magnitude),
          np.std(magnitude),
          np.min(magnitude),
          np.max(magnitude),
          np.median(magnitude),
          np.percentile(magnitude, 25),
          np.percentile(magnitude, 75),
          np.max(magnitude) - np.min(magnitude),
          # np.sum(magnitude**2) / len(magnitude),
          np.mean(np.diff(magnitude)),
          np.std(np.diff(magnitude)),

          np.corrcoef(X, Y)[0, 1],               
          np.corrcoef(X, Z)[0, 1],                
          np.corrcoef(Y, Z)[0, 1],

          #===================================
          np.mean(X),
          np.std(X),
          np.min(X),
          np.max(X),
          np.median(X),
          np.percentile(X, 25),
          np.percentile(X, 75),
          np.max(X) - np.min(X),
          # np.sum(X**2) / len(X),
          np.mean(np.diff(X)),
          np.std(np.diff(X)),

          np.mean(Y),
          np.std(Y),
          np.min(Y),
          np.max(Y),
          np.median(Y),
          np.percentile(Y, 25),
          np.percentile(Y, 75),
          np.max(Y) - np.min(Y),
          # np.sum(Y**2) / len(Y),
          np.mean(np.diff(Y)),
          np.std(np.diff(Y)),

          np.mean(Z),
          np.std(Z),
          np.min(Z),
          np.max(Z),
          np.median(Z),
          np.percentile(Z, 25),
          np.percentile(Z, 75),
          np.max(Z) - np.min(Z),
          # np.sum(Z**2) / len(Z),
          np.mean(np.diff(Z)),
          np.std(np.diff(Z))
        ])

        targets.append(0)
      
      for i in range(0, len(df_squat) - window_sample_size + 1, gap):
        window = df_squat.iloc[i : i + window_sample_size]  # sliding window

        magnitude = window["magnitude"]
        X = window["acc_x"]
        Y = window["acc_y"]
        Z = window["acc_z"]

        features.append([
          np.mean(magnitude),
          np.std(magnitude),
          np.min(magnitude),
          np.max(magnitude),
          np.median(magnitude),
          np.percentile(magnitude, 25),
          np.percentile(magnitude, 75),
          np.max(magnitude) - np.min(magnitude),
          # np.sum(magnitude**2) / len(magnitude),
          np.mean(np.diff(magnitude)),
          np.std(np.diff(magnitude)),
          np.corrcoef(X, Y)[0, 1],               
          np.corrcoef(X, Z)[0, 1],                
          np.corrcoef(Y, Z)[0, 1],

          #===================================
          np.mean(X),
          np.std(X),
          np.min(X),
          np.max(X),
          np.median(X),
          np.percentile(X, 25),
          np.percentile(X, 75),
          np.max(X) - np.min(X),
          # np.sum(X**2) / len(X),
          np.mean(np.diff(X)),
          np.std(np.diff(X)),

          np.mean(Y),
          np.std(Y),
          np.min(Y),
          np.max(Y),
          np.median(Y),
          np.percentile(Y, 25),
          np.percentile(Y, 75),
          np.max(Y) - np.min(Y),
          # np.sum(Y**2) / len(Y),
          np.mean(np.diff(Y)),
          np.std(np.diff(Y)),

          np.mean(Z),
          np.std(Z),
          np.min(Z),
          np.max(Z),
          np.median(Z),
          np.percentile(Z, 25),
          np.percentile(Z, 75),
          np.max(Z) - np.min(Z),
          # np.sum(Z**2) / len(Z),
          np.mean(np.diff(Z)),
          np.std(np.diff(Z)) 
        ])

        targets.append(1)

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

  for person in range(6, 8):
    for session in range(1, 6):
      df_step = pd.read_csv(f"data/model_data/g_force/test/{person}_step{session}_acc.csv")
      df_squat = pd.read_csv(f"data/model_data/g_force/test/{person}_squat{session}_acc.csv")

      for i in range(0, len(df_step) - window_sample_size + 1, gap):
        window = df_step.iloc[i : i + window_sample_size]  # sliding window

        magnitude = window["magnitude"]
        X = window["acc_x"]
        Y = window["acc_y"]
        Z = window["acc_z"]

        features.append([
          np.mean(magnitude),
          np.std(magnitude),
          np.min(magnitude),
          np.max(magnitude),
          np.median(magnitude),
          np.percentile(magnitude, 25),
          np.percentile(magnitude, 75),
          np.max(magnitude) - np.min(magnitude),
          # np.sum(magnitude**2) / len(magnitude),
          np.mean(np.diff(magnitude)),
          np.std(np.diff(magnitude)),
          np.corrcoef(X, Y)[0, 1],               
          np.corrcoef(X, Z)[0, 1],                
          np.corrcoef(Y, Z)[0, 1],

          #===================================
          np.mean(X),
          np.std(X),
          np.min(X),
          np.max(X),
          np.median(X),
          np.percentile(X, 25),
          np.percentile(X, 75),
          np.max(X) - np.min(X),
          # np.sum(X**2) / len(X),
          np.mean(np.diff(X)),
          np.std(np.diff(X)),

          np.mean(Y),
          np.std(Y),
          np.min(Y),
          np.max(Y),
          np.median(Y),
          np.percentile(Y, 25),
          np.percentile(Y, 75),
          np.max(Y) - np.min(Y),
          # np.sum(Y**2) / len(Y),
          np.mean(np.diff(Y)),
          np.std(np.diff(Y)),

          np.mean(Z),
          np.std(Z),
          np.min(Z),
          np.max(Z),
          np.median(Z),
          np.percentile(Z, 25),
          np.percentile(Z, 75),
          np.max(Z) - np.min(Z),
          # np.sum(Z**2) / len(Z),
          np.mean(np.diff(Z)),
          np.std(np.diff(Z)) 
        ])

        targets.append(0)
      
      for i in range(0, len(df_squat) - window_sample_size + 1, gap):
        window = df_squat.iloc[i : i + window_sample_size]  # sliding window

        magnitude = window["magnitude"]
        X = window["acc_x"]
        Y = window["acc_y"]
        Z = window["acc_z"]

        features.append([
          np.mean(magnitude),
          np.std(magnitude),
          np.min(magnitude),
          np.max(magnitude),
          np.median(magnitude),
          np.percentile(magnitude, 25),
          np.percentile(magnitude, 75),
          np.max(magnitude) - np.min(magnitude),
          # np.sum(magnitude**2) / len(magnitude),
          np.mean(np.diff(magnitude)),
          np.std(np.diff(magnitude)),
          np.corrcoef(X, Y)[0, 1],               
          np.corrcoef(X, Z)[0, 1],                
          np.corrcoef(Y, Z)[0, 1],

          #===================================
          np.mean(X),
          np.std(X),
          np.min(X),
          np.max(X),
          np.median(X),
          np.percentile(X, 25),
          np.percentile(X, 75),
          np.max(X) - np.min(X),
          # np.sum(X**2) / len(X),
          np.mean(np.diff(X)),
          np.std(np.diff(X)),

          np.mean(Y),
          np.std(Y),
          np.min(Y),
          np.max(Y),
          np.median(Y),
          np.percentile(Y, 25),
          np.percentile(Y, 75),
          np.max(Y) - np.min(Y),
          # np.sum(Y**2) / len(Y),
          np.mean(np.diff(Y)),
          np.std(np.diff(Y)),

          np.mean(Z),
          np.std(Z),
          np.min(Z),
          np.max(Z),
          np.median(Z),
          np.percentile(Z, 25),
          np.percentile(Z, 75),
          np.max(Z) - np.min(Z),
          # np.sum(Z**2) / len(Z),
          np.mean(np.diff(Z)),
          np.std(np.diff(Z)) 
        ])

        targets.append(1)

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
