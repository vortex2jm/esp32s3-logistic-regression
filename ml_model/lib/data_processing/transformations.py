import numpy as np
import pandas as pd


def raw_to_gforce(resolution, justified_bits, scale, df):
  corrected_resolution = resolution - 1  # positive and negative sides (10 bits => -512 to 511 => 2^9)
  df[['acc_x', 'acc_y', 'acc_z']] = df[['acc_x', 'acc_y', 'acc_z']] / (2**justified_bits)  # shift
  df[['acc_x', 'acc_y', 'acc_z']] = (df[['acc_x', 'acc_y', 'acc_z']] / 2**corrected_resolution) * scale  # scaling
  return df


def slice_time_serie(samples, df):
  df = df.iloc[samples:].reset_index(drop=True)
  return df


def magnitude_gen(df):
  df["magnitude"] = np.sqrt(df["acc_x"]**2 + df["acc_y"]**2 + df["acc_z"]**2)
  return df


def drop_nan(features, targets):
  features = np.array(features)
  targets = np.array(targets)
  mask = ~np.isnan(features).any(axis=1)
  features = features[mask]
  targets = targets[mask]
  return (features, targets)
