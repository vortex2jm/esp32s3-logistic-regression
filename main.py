import pandas as pd
from lib.utils import path_gen, save_to_csv
from lib.data_processing.parser import load_mat_as_dataframe
from lib.data_processing.filtering import apply_fir_filter
from lib.data_processing.transformations import slice_time_serie, raw_to_gforce, magnitude_gen
from lib.visualization.plotter import three_axis_time_signal_plot

def main():

  # First pipeline============================
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
      save_to_csv(step_df, squat_df, "./data/model_data/raw", division, folder_index, file_index)


      # FIR filter=====================================
      step_df = apply_fir_filter(400, 2, 200, step_df)
      squat_df = apply_fir_filter(400, 2, 200, squat_df)

      # Checkpoint 2 => Filtered data
      save_to_csv(step_df, squat_df, "./data/model_data/raw_filtered", division, folder_index, file_index)


      # Removing first second (outliers)==================
      step_df = slice_time_serie(samples=400, df=step_df)
      squat_df = slice_time_serie(samples=400, df=squat_df)

      # Converting to G force
      step_df = raw_to_gforce(resolution=10, justified_bits=6, scale=2, df=step_df)
      squat_df = raw_to_gforce(resolution=10, justified_bits=6, scale=2, df=squat_df)

      # Generating magnitude column
      step_df = magnitude_gen(step_df)
      step_df = magnitude_gen(step_df)

      # Checkoint 3 => G_force data
      save_to_csv(step_df, squat_df, "./data/model_data/g_force", division, folder_index, file_index)


  # three_axis_time_signal_plot(step_df, "Step Raw Data", "Time(s)", "Raw", "./graphs/ex1")
  # three_axis_time_signal_plot(step_df, "Step G-Force", "Time(s)", "Acceleration(g)", "./graphs/ex2")

if __name__ == '__main__':
  main()
