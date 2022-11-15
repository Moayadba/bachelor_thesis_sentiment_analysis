import pandas as pd
import json
from utils.data_prcessing import prepare_df


training_equal_dist = pd.read_csv("/Users/baset/Desktop/Kursanis Thesis/Datasets/manually evalueated/GME_manually_evaluated_complete_equal_dist.csv")
manually_evaluated_complete = pd.read_csv("/Users/baset/Desktop/Kursanis Thesis/Datasets/manually evalueated/GME_manually_evaluated_complete_all_sentiments.csv")

training_equal_dist_ids = training_equal_dist[["id"]]

df_validation = manually_evaluated_complete.merge(training_equal_dist_ids, how = 'outer' , indicator=True, on=['id']).loc[lambda x : x['_merge']=='left_only']

df_validation = df_validation.loc[df_validation['sentiment'].notna()]
df_validation = df_validation[['id', 'Ticker', 'full_text', 'text_processed', 'sentiment']]

positive = df_validation.loc[df_validation['sentiment'] == 'POSITIVE']
negative = df_validation.loc[df_validation['sentiment'] == 'NEGATIVE']
neutral = df_validation.loc[df_validation['sentiment'] == 'NEUTRAL']

df_validation.to_csv("/Users/baset/Desktop/Kursanis Thesis/Datasets/validatation/GME_not_used_in_training.csv")
