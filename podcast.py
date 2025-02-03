import requests
import os
from langchain_groq import ChatGroq
import time

user_id = "ssFAgjpDO5NVuWyC9AOLghsnTlS2"
secret_key = "ak-f16737b738ea4dd78ea3aefa81fb9c25"

headers = {
    'X-USER-ID': user_id,
    'Authorization': f'Bearer {secret_key}',
    'Content-Type': 'application/json',
}

model = 'PlayDialog'

#all voices here: https://docs.play.ai/tts-api-reference/voices
voice_1 = 's3://voice-cloning-zero-shot/baf1ef41-36b6-428c-9bdf-50ba54682bd8/original/manifest.json'
voice_2 = 's3://voice-cloning-zero-shot/e040bd1b-f190-4bdb-83f0-75ef85b18f84/original/manifest.json'

# Podcast transcript should be in the format of Host 1: ... Host 2 or Speaker 1 ... Speaker 2:
transcript = """
Host 1: Welcome to The Tech Tomorrow Podcast! Today we're diving into the fascinating world of voice AI and what the future holds.
Host 2: And what a topic this is. The technology has come so far from those early days of basic voice commands.
Host 1: Remember when we thought it was revolutionary just to ask our phones to set a timer?
Host 2: Now we're having full conversations with AI that can understand context, emotion, and even cultural nuances. It's incredible.
Host 1: Though it does raise some interesting questions about privacy and ethics. Where do we draw the line?
Host 2: Exactly. The potential benefits for accessibility and education are huge, but we need to be thoughtful about implementation.
Host 1: Well, we'll be exploring all of these aspects today. Stay with us as we break down the future of voice AI.
"""

payload = {
    'model': model,
    'text': transcript,
    'voice': voice_1,
    'voice2': voice_2,
    'turnPrefix': 'Host 1:',
    'turnPrefix2': 'Host 2:',
    'outputFormat': 'mp3',
}

response = requests.post('https://api.play.ai/api/v1/tts/', headers=headers, json=payload)

if response.status_code == 201:
    #job id dekhna hai to check the status
    job_id = response.json().get('id')
    url = f'https://api.play.ai/api/v1/tts/{job_id}'
    delay_seconds = 5

    # Keep checking until status is COMPLETED.
    # bohot time lagega.
    while True:
        response = requests.get(url, headers=headers)

        if response.ok:
            status = response.json().get('output', {}).get('status')
            print(f"Current Status: {status}")
            if status == 'COMPLETED':
                #audio url print hoga amazon s3 mese
                podcast_audio = response.json().get('output', {}).get('url')
                print("Podcast audio URL:", podcast_audio)
                break
            elif status == 'FAILED':
                print("Podcast generation failed.")
                break
        else:
            print("Error:", response.status_code, response.text)
            break

        time.sleep(delay_seconds)
else:
    print("Failed to initiate podcast generation:", response.status_code, response.text)
