import os
import pandas as pd
from dotenv import load_dotenv
import requests
import openai
import json
import time
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ORG_ID = os.getenv('ORG_ID')
model = "ada:ft-tu-berlin:ada-fine-tuned-with-training1-100-2022-10-09-08-19-55"
openai.organization = ORG_ID
openai.api_key = OPENAI_API_KEY
prompt_ending = "-> \n\n###\n\n "
df = pd.read_excel("/Users/baset/Desktop/sample_5_new.xlsx")
#df_test = df[101:201]
result_df = pd.DataFrame()
for i, row in df.iterrows():
    prompt_limited = row['full_text'][:2035] if len(row['full_text']) > 2049 else row['full_text']
    prompt = prompt_limited + prompt_ending
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        max_tokens=1)

    model_sentiment = response.choices[0].text
    row_eval = row[["id", "full_text", "text_processed", "sentiment"]]
    row_eval["model_sentiment"] = model_sentiment
    result_df = result_df.append(row_eval, ignore_index=True)
    print("sleep 5")
    time.sleep(2)
    print("This message will be printed after a wait of 2 seconds")
result_df.to_csv("/Users/baset/Downloads/sample_5_new_prediction_101_200_gpt_ada.csv")