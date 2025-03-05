from scipy.signal import firwin, lfilter

def apply_fir_filter(sample_rate, cut_frequency, order, df):
  fir_coeff = firwin(order, cut_frequency, fs=sample_rate, pass_zero=True)
  # Aplicando o filtro ao eixo X, Y e Z
  df["acc_x"] = lfilter(fir_coeff, 1.0, df["acc_x"])
  df["acc_y"] = lfilter(fir_coeff, 1.0, df["acc_y"])
  df["acc_z"] = lfilter(fir_coeff, 1.0, df["acc_z"])

  return df
