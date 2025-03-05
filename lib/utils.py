import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import firwin, lfilter

# Convert .mat to df====================
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



def parse_data():
  for folder_index in range(1, 8):
    for file_index in range(1, 6):
      step_path = f"./data/dataset/S{folder_index}/step{file_index}_acc.mat"
      squat_path = f"./data/dataset/S{folder_index}/squat{file_index}_acc.mat"

      df_step = load_mat_as_dataframe(step_path) 
      df_step.columns = ['time_stamp', 'acc_x', 'acc_y', 'acc_z']
      df_squat = load_mat_as_dataframe(squat_path)
      df_squat.columns = ['time_stamp', 'acc_x', 'acc_y', 'acc_z']

      if folder_index < 6:
        df_step.to_csv(f"data/model_data/train/{folder_index}_step{file_index}_acc.csv", index=False, encoding='utf-8')
        df_squat.to_csv(f"data/model_data/train/{folder_index}_squat{file_index}_acc.csv", index=False, encoding='utf-8')
        continue
      
      df_step.to_csv(f"data/model_data/test/{folder_index}_step{file_index}_acc.csv", index=False, encoding='utf-8')
      df_squat.to_csv(f"data/model_data/test/{folder_index}_squat{file_index}_acc.csv", index=False, encoding='utf-8')


def gen_example_graphs():
    # Showing raw data examples===========================================
  df_step_example = pd.read_csv("data/model_data/raw/train/1_step2_acc.csv")
  df_squat_example = pd.read_csv("data/model_data/raw/train/1_squat2_acc.csv")
  fs = 400  # Sample rate

  # Figure settings
  fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(12, 20))

  # Step graph example
  time = df_step_example['time_stamp']
  acc_x = df_step_example['acc_x']
  acc_y = df_step_example['acc_y']
  acc_z = df_step_example['acc_z']
  axes[0].plot(time, acc_x, label="Acc X", alpha=0.7)
  axes[0].plot(time, acc_y, label="Acc Y", alpha=0.7)
  axes[0].plot(time, acc_z, label="Acc Z", alpha=0.7)
  axes[0].set_ylabel("Raw")
  axes[0].set_xlabel("Time (s)")
  axes[0].set_title("Step Time Signal")
  axes[0].legend()
  axes[0].grid()
  

  acc_x_detrended = acc_x - np.mean(acc_x)  # Removing mean (DC)
  N = len(acc_x_detrended)                  # Sample amount
  frequencies = fftfreq(N, 1/fs)
  fft_x = fft(acc_x_detrended.values)       # X axis FFT 
  positive_freqs = frequencies[:N//2]       # Only positive freqs
  fft_x_mag = np.abs(fft_x)[:N//2]          # FFT's magnitude

  # Step frequency response
  axes[1].plot(positive_freqs, fft_x_mag, label='Eixo X', color='blue')
  axes[1].set_title("FFT Step X axis")
  axes[1].set_xlabel("Frequency (Hz)")
  axes[1].set_ylabel("Magnitude")
  axes[1].grid(True)
  axes[1].legend()
  axes[1].set_xticks(np.arange(1, 200, 10))
  axes[1].set_xscale('log')
  axes[1].set_xlim([1, 200])


  # Squat time response
  time = df_squat_example['time_stamp']
  acc_x = df_squat_example['acc_x']
  acc_y = df_squat_example['acc_y']
  acc_z = df_squat_example['acc_z']
  axes[2].plot(time, acc_x, label="Acc X", alpha=0.7)
  axes[2].plot(time, acc_y, label="Acc Y", alpha=0.7)
  axes[2].plot(time, acc_z, label="Acc Z", alpha=0.7)
  axes[2].set_ylabel("Raw")
  axes[2].set_xlabel("Time (s)")
  axes[2].set_title("Squat Time Signal")
  axes[2].legend()
  axes[2].grid()

  acc_x_detrended = acc_x - np.mean(acc_x)  # Removing mean (DC)
  N = len(acc_x_detrended)                  # Sample amount
  frequencies = fftfreq(N, 1/fs)
  fft_x = fft(acc_x_detrended.values)       # X axis FFT 
  positive_freqs = frequencies[:N//2]       # Only positive freqs
  fft_x_mag = np.abs(fft_x)[:N//2]          # FFT's magnitude

  # Step frequency response
  axes[3].plot(positive_freqs, fft_x_mag, label='Eixo X', color='blue')
  axes[3].set_title("FFT Squat X axis")
  axes[3].set_xlabel("Frequency (Hz)")
  axes[3].set_ylabel("Magnitude")
  axes[3].grid(True)
  axes[3].legend()
  axes[3].set_xticks(np.arange(3, 200, 10))
  axes[3].set_xscale('log')
  axes[3].set_xlim([1, 200])

  # Figure settings
  plt.tight_layout()
  plt.savefig("graphs/examples/raw_data.svg")
  plt.clf()


def fir_pass():
  # Parâmetros do filtro
  fs = 400  # Taxa de amostragem do acelerômetro (Hz)
  fc = 2  # Frequência de corte do filtro (Hz)
  N = 200 # Ordem do filtro

  # Criando o filtro FIR passa-baixa
  fir_coeff = firwin(N, fc, fs=fs, pass_zero=True)
  
  for folder_index in range(1, 8):
    for file_index in range(1, 6):
      
      if folder_index < 6:
        step_path = f"./data/model_data/raw/train/{folder_index}_step{file_index}_acc.csv"
        squat_path = f"./data/model_data/raw/train/{folder_index}_squat{file_index}_acc.csv"
      else:
        step_path = f"./data/model_data/raw/test/{folder_index}_step{file_index}_acc.csv"
        squat_path = f"./data/model_data/raw/test/{folder_index}_squat{file_index}_acc.csv"

      df_step = pd.read_csv(step_path) 
      df_squat = pd.read_csv(squat_path)
      
      # Aplicando o filtro ao eixo X, Y e Z
      df_step["acc_x"] = lfilter(fir_coeff, 1.0, df_step["acc_x"])
      df_step["acc_y"] = lfilter(fir_coeff, 1.0, df_step["acc_y"])
      df_step["acc_z"] = lfilter(fir_coeff, 1.0, df_step["acc_z"])
      df_squat["acc_x"] = lfilter(fir_coeff, 1.0, df_squat["acc_x"])
      df_squat["acc_y"] = lfilter(fir_coeff, 1.0, df_squat["acc_y"])
      df_squat["acc_z"] = lfilter(fir_coeff, 1.0, df_squat["acc_z"])
      
      df_step[['acc_x', 'acc_y', 'acc_z']] = df_step[['acc_x', 'acc_y', 'acc_z']] / (2**6)  # shift << 6
      df_step[['acc_x', 'acc_y', 'acc_z']] = (df_step[['acc_x', 'acc_y', 'acc_z']] / 2**9) * 2  # scaling
      df_step = df_step.iloc[400:].reset_index(drop=True)
      df_step["magnitude"] = np.sqrt(df_step["acc_x"]**2 + df_step["acc_y"]**2 + df_step["acc_z"]**2)
      
      df_squat[['acc_x', 'acc_y', 'acc_z']] = df_squat[['acc_x', 'acc_y', 'acc_z']] / (2**6)  # shift << 6
      df_squat[['acc_x', 'acc_y', 'acc_z']] = (df_squat[['acc_x', 'acc_y', 'acc_z']] / 2**9) * 2  # scaling
      df_squat = df_squat.iloc[400:].reset_index(drop=True)
      df_squat["magnitude"] = np.sqrt(df_squat["acc_x"]**2 + df_squat["acc_y"]**2 + df_squat["acc_z"]**2)

      if folder_index < 6:
        df_step.to_csv(f"data/model_data/g_force/train/{folder_index}_step{file_index}_acc.csv", index=False, encoding='utf-8')
        df_squat.to_csv(f"data/model_data/g_force/train/{folder_index}_squat{file_index}_acc.csv", index=False, encoding='utf-8')
      else:
        df_step.to_csv(f"data/model_data/g_force/test/{folder_index}_step{file_index}_acc.csv", index=False, encoding='utf-8')
        df_squat.to_csv(f"data/model_data/g_force/test/{folder_index}_squat{file_index}_acc.csv", index=False, encoding='utf-8')
