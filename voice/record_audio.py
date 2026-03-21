import sounddevice as sd 
from scipy.io.wavfile import write

SAMPLE_RATE = 44100
DURATION=60
CHANNELS = 1

def record_chunk(filename):
    print("STARTED RECORDING")
    audio = sd.rec(
        int(DURATION*SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
    )
    sd.wait()
    write(filename,SAMPLE_RATE,audio)
    print(f"Saved: {filename}")

i = 0
while True:
    record_chunk(f"chunk_{i}.wav")
    i = i + 1
