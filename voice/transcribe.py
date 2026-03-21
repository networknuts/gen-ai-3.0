from openai import OpenAI 
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

def transcribe(audio_file_path):
    """
    Sends an audio file to openai gpt-4.0 transcribe model
    and returns the converted text of the file.
    """
    with open(audio_file_path,"rb") as audio_file:
        response = client.audio.transcriptions.create(
            file=audio_file,
            model="gpt-4o-transcribe",
        )
        return response.text
    
audio_file_path = "chunk_0.wav"
text = transcribe(audio_file_path)
f = open("meeting-raw-notes.txt","w")
f.write(text)
f.close()
