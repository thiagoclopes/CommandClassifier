import pyaudio
import struct
import numpy as np
import time
from IPython.display import clear_output
from predict import predict
import tensorflow as tf
import sys
import serial
import struct

CHUNK = 2048
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
THRESHOLD = 350000
RECORD_SECONDS = 2

#ser = serial.Serial('COM5', 9600, timeout=0.2)
#if not ser.isOpen():
#    ser.open()


#def posicao_braco(ser, angle_base, angle_garra):
#    # ser.write(struct.pack('BBBB', angle_base, angle_garra, angle_haste1, angle_haste2))
#    ser.write(struct.pack('BB', angle_base, angle_garra))
def posicao_braco(angle_base, angle_garra):
    print(f'angulo da base:{angle_base}')
    print(f'angulo da garra:{angle_garra}')

def detecta_atividade_voz():
    global start_time
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Ouvindo...")
    gravando = False
    audio_data = b""
    while True:
        data = stream.read(CHUNK)
        fmt = f"{CHUNK}h"
        data_int = np.array(struct.unpack(fmt, data))

        # Calcula a energia do sinal de áudio
        energia = abs((np.sum(data_int ** 2) / len(data_int)))
        # print(energia)
        # Verifica se a energia é maior que o limiar
        if energia > THRESHOLD:
            print("Atividade de voz detectada")
            gravando = True
            start_time = time.time()
        if gravando:
            audio_data += data
        if gravando and time.time() - start_time > RECORD_SECONDS:
            print("Silêncio")
            yield audio_data
            gravando = False
            audio_data = b""
            clear_output(wait=True)
            # command =
            # print(command)
    stream.stop_stream()
    stream.close()
    p.terminate()


model = tf.keras.models.load_model('modelv2.keras')
angle_base = 0
angle_garra = 10
if __name__ == '__main__':
    for atividade in detecta_atividade_voz():
        command = predict(model, atividade, 2, CHANNELS, RATE)
        print(command)
        # print(atividade)
        if command == "open":
            angle_garra = 120
        elif command == "close":
            angle_garra = 10
        elif command == "right":
            angle_base = 180
        elif command == "left":
            angle_base = 0

        #posicao_braco(ser, angle_base, angle_garra)
        posicao_braco(angle_base, angle_garra)
