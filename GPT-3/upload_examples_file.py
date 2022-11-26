import os
from dotenv import load_dotenv
import requests

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
url = "https://api.openai.com/v1/models"

headers = {
    'Authorization': "Bearer {}".format(api_key)
}

files = {
    'purpose': (None, 'classifications'),
    'file': open('output.jsonl', 'rb'),
}

response = requests.post('https://api.openai.com/v1/files', headers=headers, files=files)

r_json = response.json()

print(r_json)