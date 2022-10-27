import os
from dotenv import load_dotenv
import requests

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ORG_ID = os.getenv('ORG_ID')

url = "https://api.openai.com/v1/models"

headers = {
    'Authorization': 'Bearer {}'.format(OPENAI_API_KEY),
    'OpenAI-Organization': '{}'.format(ORG_ID),
}

response = requests.get('https://api.openai.com/v1/models', headers=headers)


print("here")