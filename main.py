import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from lib.utils import *
from sklearn.preprocessing import StandardScaler

def main():
  # parse_data()
  # gen_example_graphs()
  # fir_pass()

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
          np.corrcoef(Y, Z)[0, 1] 
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
          np.corrcoef(Y, Z)[0, 1] 
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
      'C': [0.01, 0.1, 1, 10, 100, 1000],  # Regularização
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
          np.corrcoef(Y, Z)[0, 1] 
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
          np.corrcoef(Y, Z)[0, 1] 
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



if __name__ == '__main__':
  main()
