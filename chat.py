from dotenv import load_dotenv
import os
from groq import Groq
import sounddevice as sd
from gtts import gTTS
from playsound import playsound
import time
import webrtcvad
import numpy as np
from scipy.io.wavfile import write
import queue

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def speak(text):
    filename = f"reply_{int(time.time())}.mp3"
    tts = gTTS(text=text, lang='en')
    tts.save(filename)
    playsound(filename)

vad = webrtcvad.Vad(2)

def listen():
    sample_rate = 16000
    frame_duration = 30
    frame_size = int(sample_rate * frame_duration / 1000)
    max_silence_frames = 35
    
    print("Bolna shuru karo...")
    recorded_frames = []
    silence_count = 0
    speech_started = False
    
    audio_queue = queue.Queue()
    
    def callback(indata, frames, time_info, status):
        audio_queue.put(indata.copy())
    
    with sd.InputStream(samplerate=sample_rate, channels=1, dtype='int16',
                         blocksize=frame_size, callback=callback):
        while True:
            frame = audio_queue.get()
            recorded_frames.append(frame)
            
            is_speech = vad.is_speech(frame.tobytes(), sample_rate)
            
            if is_speech:
                speech_started = True
                silence_count = 0
            else:
                silence_count += 1
            
            if speech_started and silence_count >= max_silence_frames:
                break
    
    audio = np.concatenate(recorded_frames)
    write("temp.wav", sample_rate, audio)
    
    with open("temp.wav", "rb") as file:
        transcription = groq_client.audio.transcriptions.create(
            file=file,
            model="whisper-large-v3-turbo"
        )
    return transcription.text

messages = [
   {"role": "system", 
    "content": "You are a voice assistant for farmers on a marketplace app. Farmers are often busy or in the field, so keep every response to 1-2 short sentences maximum. Give direct, actionable answers first — no lengthy explanations unless the farmer specifically asks for more details. Avoid lists, avoid markdown, speak like a helpful local shopkeeper giving quick advice."}
]
while True:
    
    user_input = listen()
    print("Tum bole:", user_input)

    if not user_input or len(user_input.strip()) < 3:
       print("kuch sunayi nhi diya , phir se bolo....")
       continue


    if user_input.lower().strip().rstrip(".") in ["exit", "bye", "goodbye"]:
        break
    
    messages.append({"role": "user", "content": user_input})
    
    response = groq_client.chat.completions.create(
        messages=messages,
        model="openai/gpt-oss-20b",
        max_tokens = 200
    )
    
    reply = response.choices[0].message.content
    if not reply:
        reply = "Sorry, could you repeat that please ?"
    print("Assistant:", reply)
    speak(reply)
    
    messages.append({"role": "assistant", "content": reply})
    
