import scipy.io
import numpy as np
import pandas as pd


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
