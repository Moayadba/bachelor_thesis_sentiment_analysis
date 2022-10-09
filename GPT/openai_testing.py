import os
from dotenv import load_dotenv
import requests

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
url = "https://api.openai.com/v1/models"

headers = {
    # Already added when you pass json= but not when you pass data=
    # 'Content-Type': 'application/json',
    'Authorization': "Bearer {}".format(api_key)
}

json_data = {
    'model': 'text-davinci-002',
    'prompt': 'Say this is a test',
    'temperature': 0,
    'max_tokens': 6,
}

response = requests.post('https://api.openai.com/v1/completions', headers=headers, json=json_data)

print("here")