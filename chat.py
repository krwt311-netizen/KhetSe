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
import json
import requests

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
            model="whisper-large-v3-turbo",
            language="hi"
        )
    return transcription.text

tools = [
    {
        "type": "function",
        "function": {
            "name": "create_listing",
            "description": "Create a crop listing when the farmer wants to sell their produce",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {"type": "string", "description": "Name of the crop, e.g. tomato, wheat"},
                    "quantity": {"type": "number", "description": "Quantity available"},
                    "unit": {"type": "string", "enum": ["kg", "tonnes", "quintal"]},
                    "asking_price": {"type": "number", "description": "Price farmer wants per unit"},
                    "price_unit": {"type": "string", "description": "Usually same as unit, e.g. kg"},
                    "location": {"type": "string", "description": "Farmer's location or nearby city"},
                    "quality": {"type": "string", "description": "Quality grade if mentioned"}
                },
                "required": ["product", "quantity", "location"]
            }
        }
    }
]

def send_to_backend(data):
    payload = {
        "farmer_name": "Demo Farmer",
        "phone": "+919999999999",
        "intent": "sell",
        "source": "voice_agent",
        **data
    }
    try:
        response = requests.post("http://localhost:8787/api/listings", json=payload)
        return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}

messages = [
    {"role": "system",
     "content": "You are a voice assistant for farmers on a marketplace app. Keep responses to 1-2 short sentences. "
                "When a farmer wants to sell produce, use the create_listing function — but only call it once you "
                "know the product, quantity, unit, asking price, and location. If any of these are missing, ask the "
                "farmer for the missing detail instead of guessing. Avoid lists, avoid markdown."}
]

while True:
    user_input = listen()
    print("Tum bole:", user_input)

    if not user_input or len(user_input.strip()) < 3:
        print("kuch sunayi nhi diya, phir se bolo....")
        continue

    if user_input.lower().strip().rstrip(".") in ["exit", "bye", "goodbye"]:
        break

    messages.append({"role": "user", "content": user_input})

    response = groq_client.chat.completions.create(
        messages=messages,
        model="openai/gpt-oss-20b",
        max_tokens=600,
        tools=tools
    )

    message = response.choices[0].message

    if message.tool_calls:
        tool_call = message.tool_calls[0]
        try:
            listing_data = json.loads(tool_call.function.arguments)

            required_fields = ["product", "quantity", "unit", "asking_price", "location"]
            missing = [f for f in required_fields if f not in listing_data or not listing_data[f]]

            if missing:
                reply = f"Please batao: {', '.join(missing)}"
            else:
                print("Extracted data:", listing_data)
                result = send_to_backend(listing_data)
                print("Backend response:", result)

                if result.get("success"):
                    reply = "Aapki listing ban gayi hai!"
                else:
                    reply = "Kuch dikkat aayi, dobara try karo."
        except json.JSONDecodeError:
            reply = "Samajh nahi paya, ek baar phir se bolo please."
    else:
        reply = message.content
        if not reply:
            reply = "Sorry, could you repeat that please?"

    print("Assistant:", reply)
    speak(reply)

    messages.append({"role": "assistant", "content": reply})