import numpy as np


def extract_metrics(df, window_width, gap_interval, sample_rate, category):
  window_sample_size = window_width * sample_rate
  gap = int(sample_rate * gap_interval)
  
  category_num = {
    "step": 0,
    "squat": 1
  }

  features = []
  targets =[]

  for i in range(0, len(df) - window_sample_size + 1, gap):
    window = df.iloc[i : i + window_sample_size]  # sliding window

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

    targets.append(category_num[category])

  return (features, targets)
