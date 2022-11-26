from finBERT.finbert.finbert import predict
from transformers import AutoModelForSequenceClassification
import argparse
import os
import pandas as pd
import datetime as dt
import sys

# path of the file with posts you want to predict the sentiment for
INPUT_FILE_PATH = '/Users/baset/Desktop/Kursanis Thesis/Datasets/validatation/GME_final_validation_sample_equal_dist_3.csv'
# path of the file with posts you want to write the predictions to
OUTPUT_FILE_PATH = '/Users/baset/Downloads/GME_final_validation_sample_equal_dist_prediction_bert.csv'
# path of the pre-trained model file
MODEL_FILE_PATH = "/Users/baset/PycharmProjects/sentiment_analysis_all_models/finBERT/models/sentiment/FinPhraseBank"


def predict_df(df):
    output_df = pd.DataFrame()
    for i, row in df.iterrows():
        text = row['full_text']
        prediction = predict(text,model,write_to_csv=False)
        row_eval = row[["id","full_text", "text_processed", "sentiment"]]
        row_eval["model_sentiment"] = prediction["prediction"][0]
        output_df = output_df.append(row_eval, ignore_index=True)

    output_df.to_csv(OUTPUT_FILE_PATH)

if __name__ == '__main__':
    start = dt.datetime.now()
    print(f'\n\n{start.strftime("%c")}\nPROGRAM NAME: {sys.argv[0]}\n')
    df = pd.read_csv(INPUT_FILE_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_FILE_PATH, num_labels=3, cache_dir=None)
    predict_df(df)
    print(f'\n\nRuntime: {(dt.datetime.now()-start)}')
    print(f'\nNormal termination.\n{dt.datetime.now().strftime("%c")}\n')
