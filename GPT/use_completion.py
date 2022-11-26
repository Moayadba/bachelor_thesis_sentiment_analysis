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
model = "babbage:ft-tu-berlin:babbage-fine-tuned-with-aapl-complete-2022-10-21-19-14-26"
#ada:ft-tu-berlin:ada-fine-tuned-with-training1-100-2022-10-09-08-19-55
#ada:ft-tu-berlin:ada-fine-tuned-with-training-1-2-5-2022-10-13-08-58-44
#ada:ft-tu-berlin:ada-fine-tuned-with-training-gme-only-2022-10-14-08-43-31
#ada:ft-tu-berlin:ada-fine-tuned-with-training-aapl-only-2022-10-14-11-45-16
#curie:ft-tu-berlin:curie-fine-tuned-with-training-aapl-only-2022-10-14-15-49-24
#curie:ft-tu-berlin:curie-fine-tuned-with-training-gme-only-2022-10-14-16-24-30
#curie:ft-tu-berlin:curie-fine-tuned-with-training-gme-only-2022-10-14-16-50-21
#babbage:ft-tu-berlin:babbage-fine-tuned-with-gme-only-2022-10-14-17-29-59
#babbage:ft-tu-berlin:babbage-fine-tuned-with-gme-complete-2022-10-16-14-24-56
#babbage:ft-tu-berlin:babbage-fine-tuned-with-gme-complete-2-2022-10-16-16-26-57
#babbage:ft-tu-berlin:babbage-fine-tuned-with-aapl-complete-2022-10-21-19-14-26
errors = []
openai.organization = ORG_ID
openai.api_key = OPENAI_API_KEY
prompt_ending = "-> \n\n###\n\n "
df = pd.read_csv("/Users/baset/Desktop/Kursanis Thesis/Datasets/validatation/GME_final_validation_sample_equal_dist_3.csv")
#df = df.rename(columns={'full_text_x': 'full_text', 'text_processed_x': 'text_processed', 'sentiment_x': 'sentiment'})
result_df = pd.DataFrame()
for i, row in df.iterrows():
    try:
        prompt_limited = row['full_text'][:2035] if len(row['full_text']) > 2049 else row['full_text']
        prompt = prompt_limited + prompt_ending
        response = openai.Completion.create(
            model=model,
            prompt=prompt,
            max_tokens=1)

        model_sentiment = response.choices[0].text
        row_eval = row[["id", "full_text", "text_processed", "sentiment"]]
        #row_eval = row[["id", "full_text", "text_processed"]]
        row_eval["model_sentiment"] = model_sentiment
        result_df = result_df.append(row_eval, ignore_index=True)
        print("processing the row num: {}",format(str(i)))
    except Exception as e:
        print('in row: {} an exception occured. message: {}'.format(str(i), e))
        errors.append('in row: {} an exception occured. message: {}'.format(str(i), e))
        time.sleep(5)
        continue

result_df.to_csv("/Users/baset/Desktop/Kursanis Thesis/Datasets/validatation/GME_final_validation_sample_equal_dist_prediction_4.csv")
#result_df.to_excel("/Users/baset/Downloads/df_1_unique_gpt_babbage_2.xlsx")