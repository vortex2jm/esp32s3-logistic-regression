from lib.data_processing.transformations import slice_time_serie, raw_to_gforce, magnitude_gen, drop_nan
from lib.data_processing.sliding_window import extract_metrics
from lib.data_processing.parser import load_mat_as_dataframe
from lib.data_processing.filtering import apply_fir_filter

from lib.visualization.plotter import three_axis_time_signal_plot

from lib.utils import path_gen, save_to_csv, save_to_parquet

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

import pandas as pd


def main():

  X_train = []
  y_train = []
  X_test = []
  y_test = []

  # Pre-processing pipeline============================
  for folder_index in range(1, 8):  # 1 => 7
    for file_index in range(1, 6):  # 1 => 5

      division = "train" if folder_index < 6 else "test"  # Train x Test division
      step_path, squat_path = path_gen(src_path="./data/dataset", folder_index= folder_index, file_index=file_index)  

      # Parsing .mat===========================
      step_df = load_mat_as_dataframe(step_path) 
      step_df.columns = ['time_stamp', 'acc_x', 'acc_y', 'acc_z']
      squat_df = load_mat_as_dataframe(squat_path)
      squat_df.columns = ['time_stamp', 'acc_x', 'acc_y', 'acc_z']

      # Checkpoint 1 => Raw data
      # save_to_csv(step_df, squat_df, "./data/model_data/raw", division, folder_index, file_index)
      save_to_parquet(step_df, squat_df, "./data/model_data/raw", division, folder_index, file_index)

      # FIR filter=====================================
      step_df = apply_fir_filter(400, 2, 200, step_df)
      squat_df = apply_fir_filter(400, 2, 200, squat_df)

      # Checkpoint 2 => Filtered data
      # save_to_csv(step_df, squat_df, "./data/model_data/raw_filtered", division, folder_index, file_index)
      save_to_parquet(step_df, squat_df, "./data/model_data/raw_filtered", division, folder_index, file_index)


      # Removing first second (outliers)==================
      step_df = slice_time_serie(samples=400, df=step_df)
      squat_df = slice_time_serie(samples=400, df=squat_df)

      # Converting to G force
      step_df = raw_to_gforce(resolution=10, justified_bits=6, scale=2, df=step_df)
      squat_df = raw_to_gforce(resolution=10, justified_bits=6, scale=2, df=squat_df)

      # Generating magnitude column
      step_df = magnitude_gen(step_df)
      squat_df = magnitude_gen(squat_df)

      # Checkoint 3 => G_force data
      # save_to_csv(step_df, squat_df, "./data/model_data/g_force", division, folder_index, file_index)
      save_to_parquet(step_df, squat_df, "./data/model_data/g_force", division, folder_index, file_index)

      # Getting metrics vectors===================================
      step_feature_matrix, step_tartget_vector = extract_metrics(       # Step
        df=step_df, 
        category="step", 
        sample_rate=400, 
        window_width=10, 
        gap_interval= 0.5
      )

      squat_feature_matrix, squat_tartget_vector = extract_metrics(     # Squat
        df=squat_df, 
        category="squat", 
        sample_rate=400, 
        window_width=10, 
        gap_interval= 0.5
      )
      
      if division == "train":
        X_train.extend([feature_vector for feature_vector in step_feature_matrix])
        X_train.extend([feature_vector for feature_vector in squat_feature_matrix])

        y_train.extend([target for target in step_tartget_vector])
        y_train.extend([target for target in squat_tartget_vector])

      else:
        X_test.extend([feature_vector for feature_vector in step_feature_matrix])
        X_test.extend([feature_vector for feature_vector in squat_feature_matrix])
        
        y_test.extend([target for target in step_tartget_vector])
        y_test.extend([target for target in squat_tartget_vector])

  
  # Generating graphs
  d_raw = pd.read_parquet("data/model_data/raw/train/1_step3_acc.parquet")
  d_filtered = pd.read_parquet("data/model_data/raw_filtered/train/1_step3_acc.parquet")
  d_gforce = pd.read_parquet("data/model_data/g_force/train/1_step3_acc.parquet")
  
  three_axis_time_signal_plot(d_raw, "Squat Raw Data", "Time(s)", "Raw", "./graphs/raw")  
  three_axis_time_signal_plot(d_filtered, "Squat Filtered Data", "Time(s)", "Raw", "./graphs/filtered")  
  three_axis_time_signal_plot(d_gforce, "Squat Gforce Data", "Time(s)", "Acceleration(g)", "./graphs/gforce")  


  # Training model==============================
  # removing NaN values
  X_train, y_train = drop_nan(X_train, y_train)
  X_test, y_test = drop_nan(X_test, y_test)

  # Normalization
  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.fit_transform(X_test)

  # # Cross validation
  param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],  # Regularização
    'penalty': ['l1', 'l2'],  # Penalização (L1 para LASSO, L2 para Ridge)
    'solver': ['liblinear']  # Necessário para suportar L1 e L2
  }

  grid_search = GridSearchCV(LogisticRegression(max_iter=5000), param_grid, n_jobs=-1)
  grid_search.fit(X_train, y_train)
  print("Melhores hiperparâmetros:", grid_search.best_params_)

  # # # Testing the best model
  best_model = grid_search.best_estimator_
  y_pred = best_model.predict(X_test)

  # weights and bias
  print("Weights")
  print(best_model.coef_)
  print("\nBias")
  print(best_model.intercept_)

  # Avaliating
  accuracy = accuracy_score(y_test, y_pred)
  conf_matrix = confusion_matrix(y_test, y_pred)
  class_report = classification_report(y_test, y_pred)

  print("\nAcurácia:", accuracy)
  print("\nMatriz de Confusão:\n", conf_matrix)
  print("\nRelatório de Classificação:\n", class_report)


if __name__ == '__main__':
  main()
