import os
import pandas as pd
from dotenv import load_dotenv
import requests
import openai
import json
import time
import sys
import datetime as dt
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ORG_ID = os.getenv('ORG_ID')

# path of the file with posts you want to predict the sentiment for
INPUT_FILE_PATH = "/Users/baset/Desktop/Kursanis Thesis/Datasets/validatation/GME_final_validation_sample_equal_dist_3.csv"
# path of the file with posts you want to write the predictions to
OUTPUT_FILE_PATH = '/Users/baset/Downloads/GME_final_validation_sample_equal_dist_prediction_GPT3.csv'
# ID of the model (as returned from the call to the fine-tuned endpoint)
MODEL_ID = "babbage:ft-tu-berlin:babbage-fine-tuned-with-188-12-2022-11-26-17-09-03"

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

def predict_df(df):
    prompt_ending = "-> \n\n###\n\n "
    result_df = pd.DataFrame()
    for i, row in df.iterrows():
        try:
            prompt_limited = row['full_text'][:2035] if len(row['full_text']) > 2049 else row['full_text']
            prompt = prompt_limited + prompt_ending
            response = openai.Completion.create(
                model=MODEL_ID,
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

    result_df.to_csv(OUTPUT_FILE_PATH)

if __name__ == '__main__':
    start = dt.datetime.now()
    print(f'\n\n{start.strftime("%c")}\nPROGRAM NAME: {sys.argv[0]}\n')
    df = pd.read_csv(INPUT_FILE_PATH)
    predict_df(df)
    print(f'\n\nRuntime: {(dt.datetime.now()-start)}')
    print(f'\nNormal termination.\n{dt.datetime.now().strftime("%c")}\n')