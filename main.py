from lib.utils import load_mat_as_dataframe

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
