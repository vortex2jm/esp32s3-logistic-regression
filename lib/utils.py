def path_gen(src_path, division, folder_index, file_index):
  step_path = f"{src_path}/{division}/{folder_index}_step{file_index}_acc.csv"
  squat_path = f"{src_path}/{division}/{folder_index}_squat{file_index}_acc.csv"
  return (step_path, squat_path)


def save_to_parquet(df_step, df_squat, dest_path, division, folder_index, file_index):
  df_step.to_parquet(f"{dest_path}/{division}/{folder_index}_step{file_index}_acc.csv", engine="pyarrow", compression="snappy")
  df_squat.to_parquet(f"{dest_path}/{division}/{folder_index}_squat{file_index}_acc.csv", engine="pyarrow", compression="snappy")


def save_to_csv(df_step, df_squat, dest_path, division, folder_index, file_index):
  df_step.to_csv(f"{dest_path}/{division}/{folder_index}_step{file_index}_acc.csv", index=False, encoding='utf-8')
  df_squat.to_csv(f"{dest_path}/{division}/{folder_index}_squat{file_index}_acc.csv", index=False, encoding='utf-8')
