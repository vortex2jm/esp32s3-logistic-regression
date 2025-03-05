import scipy.io
import numpy as np
import pandas as pd

#==========================================================================
def load_mat_as_dataframe(mat_file_path):
  # Load the .mat file
  mat_data = scipy.io.loadmat(mat_file_path)

  # Exclude meta entries (those starting with '__')
  data_keys = [key for key in mat_data.keys() if not key.startswith('__')]

  # Assume the primary dataset is stored in the first non-meta key
  if len(data_keys) == 0:
      raise ValueError("No usable data found in the .mat file.")

  # Extract data (assuming it's a table-like structure or 2D array)
  primary_data = mat_data[data_keys[0]]

  # Convert to a DataFrame if the data is structured
  if isinstance(primary_data, np.ndarray) and primary_data.ndim == 2:
      df = pd.DataFrame(primary_data)
  else:
      raise ValueError("The data is not in a tabular or array-like format.")

  if df.isna().any().all():
      print("Nan value in dataframe {mat_file_path}")
  return df


#==========================================================================
def mat_to_csv(src_path, dest_path, session_amount, measures_per_session):
  for folder_index in range(1, (session_amount + 1)):
    for file_index in range(1, (measures_per_session + 1)):
      
      # Source path
      step_path = f"{src_path}/S{folder_index}/step{file_index}_acc.mat"
      squat_path = f"{src_path}/S{folder_index}/squat{file_index}_acc.mat"
      
      # Loading datasets
      df_step = load_mat_as_dataframe(step_path) 
      df_step.columns = ['time_stamp', 'acc_x', 'acc_y', 'acc_z']
      df_squat = load_mat_as_dataframe(squat_path)
      df_squat.columns = ['time_stamp', 'acc_x', 'acc_y', 'acc_z']
      
      # Saving csv
      division = "train" if folder_index < 6 else "test"
      df_step.to_csv(f"{dest_path}/{division}/{folder_index}_step{file_index}_acc.csv", index=False, encoding='utf-8')
      df_squat.to_csv(f"{dest_path}/{division}/{folder_index}_squat{file_index}_acc.csv", index=False, encoding='utf-8')
