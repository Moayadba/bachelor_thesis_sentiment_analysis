import pandas as pd
import json
from prepare_data_sample import prepare_df

df_GME = pd.read_csv('/Users/baset/Desktop/Kursanis Thesis/Abschlussarbeit Reddit/GME_v2.csv')
df_1 = pd.read_excel('/Users/baset/Desktop/sample_1.xlsx')
df_1['Ticker'] = 'GME'
df_1 = df_1.rename(columns={'full_text_processed': 'text_processed'})
df_1 = df_1[['id', 'Ticker', 'full_text', 'text_processed', 'sentiment']]

df_2 = pd.read_excel('/Users/baset/Desktop/sample_1_unique.xlsx')

df_3 = pd.read_excel('/Users/baset/Desktop/sample_2_unique.xlsx')
df_3 = df_3[0:100]

df_4 = pd.read_csv('/Users/baset/Desktop/GME_negative.csv')
df_4 = df_4.drop('model_sentiment', axis=1)
df_4 = df_4.drop('Unnamed: 0', axis=1)
df_4 = df_4.drop('index', axis=1)
df_4['Ticker'] = 'GME'

df_5 = pd.read_excel('/Users/baset/Desktop/df_validate_GME_only_gpt_curie_4000.xlsx')
df_5 = df_5.loc[df_5["sentiment"] == 'NEGATIVE']
df_5 = df_5.drop('model_sentiment', axis=1)
df_5['Ticker'] = 'GME'

df_6 = pd.read_excel('/Users/baset/Desktop/df_validate_GME_only_gpt_curie_300_4000.xlsx')
df_6 = df_6.loc[df_6["sentiment"] == 'NEGATIVE']
df_6 = df_6.drop('model_sentiment', axis=1)
df_6['Ticker'] = 'GME'


df_GME_complete = pd.concat([df_1, df_2, df_3, df_4, df_5, df_6], ignore_index=True)
df_GME_complete_negative = df_GME_complete.loc[df_GME_complete["sentiment"] == "NEGATIVE"]
df_GME_complete_positive = (df_GME_complete.loc[df_GME_complete["sentiment"] == "POSITIVE"]).sample(n=100)
df_GME_complete_neutral = (df_GME_complete.loc[df_GME_complete["sentiment"] == "NEUTRAL"]).sample(n=100)
df_GME_complete_training = pd.concat([df_GME_complete_negative, df_GME_complete_positive, df_GME_complete_neutral], ignore_index=True)
df_GME_complete_training = df_GME_complete_training.sample(frac=1).reset_index(drop=True)


df_GME_complete_training.to_csv("/Users/baset/Desktop/GME_manually_evaluated_complete.csv")


list_exclude = list(df_GME_complete['id'])

sample_GME = prepare_df(df_GME, list_exclude, 2000)

df_GME_complete_positive = df_GME_complete.loc[df_GME_complete['sentiment'] == 'POSITIVE']
df_GME_complete_negative = df_GME_complete.loc[df_GME_complete['sentiment'] == 'NEGATIVE']
df_GME_complete_neutral = df_GME_complete.loc[df_GME_complete['sentiment'] == 'NEUTRAL']


sample_GME.to_csv("/Users/baset/Desktop/sample_GME_2000.csv")
