import pandas as pd
import redditcleaner
from prepare_data_sample import prepare_df


df = pd.read_csv('/Users/baset/Desktop/Kursanis Thesis/Abschlussarbeit Reddit/AAPL.csv')

df_processed = prepare_df(df, without_na=False)
df_processed = df_processed.fillna('')
df_processed["text_cleaned"] = df_processed['full_text'].map(redditcleaner.clean)
df_processed.to_excel("/Users/baset/Desktop/AAPL_complete_dataset_cleanedup.xlsx")
print('here')