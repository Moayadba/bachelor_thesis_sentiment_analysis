import requests
import os
from dotenv import load_dotenv
import requests
import openai
import json
import time
load_dotenv()


OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ORG_ID = os.getenv('ORG_ID')

headers = {
    'Authorization': 'Bearer {}'.format(OPENAI_API_KEY),
}
openai.organization = ORG_ID
openai.api_key = OPENAI_API_KEY
my_models = openai.FineTune.list()

response = requests.get('https://api.openai.com/v1/models', headers=headers)

print('done')