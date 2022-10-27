import os
import pandas as pd
from dotenv import load_dotenv
import requests
import openai
import json
import time

AAPL = pd.read_csv("/Users/baset/Desktop/Kursanis Thesis/Abschlussarbeit Reddit/AAPL.csv")

APPL_per_day = AAPL.groupby(['NY_Day'])['id'].count()

print('here')