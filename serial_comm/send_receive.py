import csv
import time
import serial
import threading

PORT = "COM9"
BAUDRATE = 115200
FEATURES_PATH = "./features_test.csv"

def load_features(path):
  features = []
  targets = []
  with open(path, mode='r') as file:
    reader = csv.reader(file)
    for row in reader:
      feature_vector = list(map(float, row[:-1]))
      target = int(row[-1])
      features.append(feature_vector)
      targets.append(target)
  return (features, targets)


def serial_read(ser):
  while True:
    try:
      message = ser.readline().decode("utf-8", errors="ignore").strip()
      if message:
        print(f"Received: {message}")
    except serial.SerialException as e:
      print(f"Read error: {e}")
      break

# Função para enviar dados pela porta serial
def serial_write(ser):
  
  features, _ = load_features(FEATURES_PATH)
  feature_index=0

  while True:
    # inp = input()	# Type something into terminal to send features
    # features = f"{1.39},{2.40},{3.41},{-1.20},{0.45},{2.71},{1.95},{-3.45},{1.00},{0.67},{1.34},{1.19},{-2.7}"
    # if inp:
    features_vector_string = "" # Creating empty string
    
    for value in features(feature_index):
      features_vector_string+=f"{value},"
    
    features_vector_string = features_vector_string[:-1]

    try:
      time.sleep(0.5)
      # ser.write(features_vector_string.encode("utf-8"))
      print(features_vector_string)
      feature_index+=1

    except serial.SerialException as e:
      print(f"Write error: {e}")
      break		


def main():
  try:
    with serial.Serial(PORT, BAUDRATE, timeout=1) as ser:
      print(f"Connected to port {PORT}. Waiting data...")
      
			# creating threads for non block processes
      thread_leitura = threading.Thread(target=serial_read, args=(ser,))
      thread_envio = threading.Thread(target=serial_write, args=(ser,))

			# Starting threads
      thread_leitura.start()
      thread_envio.start()
      
			# Waiting for threads end
      thread_leitura.join()
      thread_envio.join()
      
  except serial.SerialException as e:
    print(f"Serial port access error: {e}")


if __name__ == "__main__":
    main()