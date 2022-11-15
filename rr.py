import pandas as pd
import json
from utils.data_prcessing import prepare_df

df_GME = pd.read_csv('/Users/baset/Desktop/Kursanis Thesis/Abschlussarbeit Reddit/GME_v2.csv')

df_GME_already_evaluated = pd.read_csv("/Users/baset/Desktop/Kursanis Thesis/Datasets/manually evalueated/GME_manually_evaluated_complete_all_sentiments.csv")

list_exclude = list(df_GME_already_evaluated['id'])

sample = prepare_df(df_GME, list_exclude, 2000, without_na=False)

sample.to_excel("/Users/baset/Desktop/Kursanis Thesis/Datasets/manually evalueated/follow_manual_evaluation_GME.xlsx")