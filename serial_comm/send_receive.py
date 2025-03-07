import serial
import threading

PORT = "COM9"
BAUDRATE = 115200

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
  while True:
    inp = input()	# Type something into terminal to send features
    features = f"{1.39},{2.40},{3.41}"
    if inp:
      try:
        ser.write(features.encode("utf-8"))
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
