import pandas as pd
import re


df1 = pd.read_excel("/Users/baset/Desktop/sample_5_new.xlsx")
df2 = pd.read_excel("/Users/baset/Desktop/sample_2_unique.xlsx")
df3= pd.read_excel('/Users/baset/Downloads/AAPL_1000.xlsx')

training= pd.read_csv('/Users/baset/Desktop/final_training_sample_APPL.csv')


df2 = df2[100:]
df_AAPL = pd.concat([df1,df2, df3])

df_validation = df_AAPL.merge(training, how = 'outer' ,indicator=True, on=['id']).loc[lambda x : x['_merge']=='left_only']
df_validation = df_validation.loc[df_validation['sentiment_x'].notna()]
positive =df_AAPL.loc[df_AAPL['sentiment'] == 'POSITIVE']
negative = df_AAPL.loc[df_AAPL['sentiment'] == 'NEGATIVE']
neutral = df_AAPL.loc[df_AAPL['sentiment'] == 'NEUTRAL']

negative = negative.sample(n=100)
positive = positive.sample(n=100)
neutral = neutral.sample(n=100)

final_df = pd.concat([negative, positive, neutral])
final_df = final_df.sample(frac=1).reset_index(drop=True)

final_df.to_csv("/Users/baset/Desktop/final_training_sample_APPL.csv")
print('here')