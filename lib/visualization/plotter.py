from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def three_axis_time_signal_plot(df, title, x_label, y_label, dest_path):
  time = df['time_stamp']
  acc_x = df['acc_x']
  acc_y = df['acc_y']
  acc_z = df['acc_z']

  plt.figure(figsize=(12, 5))
  plt.plot(time, acc_x, label="Acc X", alpha=0.7)
  plt.plot(time, acc_y, label="Acc Y", alpha=0.7)
  plt.plot(time, acc_z, label="Acc Z", alpha=0.7)
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.title(title)
  plt.legend()
  plt.grid()
  plt.savefig(dest_path)
  plt.clf()


def magnitude_time_signal_plot(df, title, x_label, y_label, dest_path):
  time = df['time_stamp']
  magnitude = df["magnitude"]

  plt.figure(figsize=(12, 5))
  plt.plot(time, magnitude, label="Magnitude", alpha=0.7)
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.title(title)
  plt.legend()
  plt.grid()
  plt.savefig(dest_path)
  plt.clf()


def fft_plot(df, sample_rate, axis, title, x_label, y_label, dest_path):
  acc = df[axis]

  acc_detrended = acc - np.mean(acc)  # Removing mean (DC)
  N = len(acc_detrended)                  # Sample amount
  frequencies = fftfreq(N, 1/sample_rate)
  fft_x = fft(acc_detrended.values)       # X axis FFT 
  positive_freqs = frequencies[:N//2]       # Only positive freqs
  fft_x_mag = np.abs(fft_x)[:N//2]          # FFT's magnitude

  plt.figure(figsize=(12, 5))
  plt.plot(positive_freqs, fft_x_mag, label='Eixo X', color='blue')
  plt.xlabel(x_label)
  plt.xscale('log')
  plt.ylabel(y_label)
  plt.title(title)
  plt.legend()
  plt.grid()
  plt.savefig(dest_path)
  plt.clf()
