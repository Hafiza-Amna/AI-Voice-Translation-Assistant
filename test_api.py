import requests

url = "http://127.0.0.1:8000/api/v1/translate-audio"
files = {'audio_file': ('jfk.wav', open('jfk.wav', 'rb'), 'audio/wav')}
data = {'source_language': 'en', 'target_language': 'ur'}

response = requests.post(url, files=files, data=data)
print("Status Code:", response.status_code)
print("Response JSON:", response.text)
