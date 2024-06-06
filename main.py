import pyaudio
import struct
import numpy as np
import time
from IPython.display import clear_output
from predict import predict
import tensorflow as tf
import sys

CHUNK = 2048
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
THRESHOLD = 350000
RECORD_SECONDS = 2


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

if __name__ == '__main__':
    for atividade in detecta_atividade_voz():
        command = predict(model, atividade, 2, CHANNELS, RATE)
        print(command)
        # print(atividade)
        if command == "close":
            print("Encerrando o programa...")
            sys.exit()