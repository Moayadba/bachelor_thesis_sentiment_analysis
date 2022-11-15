import pandas as pd
import re


df1 = pd.read_excel("/Users/baset/Desktop/Kursanis Thesis/classfied/sample_5_new.xlsx")
df2 = pd.read_excel("/Users/baset/Desktop/Kursanis Thesis/classfied/sample_2_unique.xlsx")
df3= pd.read_excel('/Users/baset/Desktop/Kursanis Thesis/classfied/AAPL_1000.xlsx')

training= pd.read_csv('/Users/baset/Desktop/Kursanis Thesis/classfied/final_training_sample_APPL.csv')


df2 = df2.loc[df2['Ticker'] == "AAPL"]

df1 = df1.reset_index(drop=True)
df2 = df2.reset_index(drop=True)
df3 = df3.reset_index(drop=True)

df_AAPL = pd.concat([df1,df2, df3])

df_AAPL = df_AAPL.loc[df_AAPL['sentiment'].notna()]

df_AAPL = df_AAPL.reset_index(drop=True)

training_ids = training[["id"]]

df_validation = df_AAPL.merge(training_ids, how = 'outer' ,indicator=True, on=['id']).loc[lambda x : x['_merge']=='left_only']

df_validation = df_validation.loc[df_validation['sentiment'].notna()]

positive =df_validation.loc[df_validation['sentiment'] == 'POSITIVE']
negative = df_validation.loc[df_validation['sentiment'] == 'NEGATIVE']
neutral = df_validation.loc[df_validation['sentiment'] == 'NEUTRAL']

df_validation = df_validation[['id', 'Ticker', 'full_text', 'text_processed', 'sentiment']]

df_validation.to_csv("/Users/baset/Desktop/Kursanis Thesis/Datasets/validatation/AAPL_not_used_in_training.csv")

###############################

negative = negative.sample(n=100)
positive = positive.sample(n=100)
neutral = neutral.sample(n=100)

final_df = pd.concat([negative, positive, neutral])
final_df = final_df.sample(frac=1).reset_index(drop=True)

final_df.to_csv("/Users/baset/Desktop/final_training_sample_APPL.csv")
print('here')