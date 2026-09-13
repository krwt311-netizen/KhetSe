import sounddevice as sd
from scipy.io.wavfile import write
from dotenv import load_dotenv
import os
from groq import Groq

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

duration = 5
sample_rate = 16000

print("Recording... bolna shuru karo")
audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
sd.wait()

write("test_output.wav", sample_rate, audio)
print("Recording saved")

# Ab Whisper ko bhejo transcription ke liye
with open("test_output.wav", "rb") as file:
    transcription = groq_client.audio.transcriptions.create(
        file=file,
        model="whisper-large-v3"
    )

print("Tumne bola:", transcription.text)