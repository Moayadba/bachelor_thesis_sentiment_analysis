from finbert.finbert import predict
from transformers import AutoModelForSequenceClassification
import argparse
import os
import pandas as pd

df = pd.read_csv("/Users/baset/Desktop/Kursanis Thesis/Datasets/validatation/GME_final_validation_sample_equal_dist_3.csv")
model_path = "/Users/baset/PycharmProjects/finBERT/models/sentiment/TRC2"
model = AutoModelForSequenceClassification.from_pretrained(model_path,num_labels=3,cache_dir=None)


print('here')
result_df = pd.DataFrame()

for i, row in df.iterrows():
    text = row['full_text']
    prediction = predict(text,model,write_to_csv=False)
    row_eval = row[["id","full_text", "text_processed", "sentiment"]]
    row_eval["model_sentiment"] = prediction["prediction"][0]
    result_df = result_df.append(row_eval, ignore_index=True)

result_df.to_csv("/Users/baset/Desktop/Kursanis Thesis/Datasets/validatation/GME_final_validation_sample_equal_dist_3_prediction_BERT_2.csv")
