import jsonlines
import pandas as pd

df = pd.read_excel("/Users/baset/Desktop/sample_1.xlsx")
training_df = pd.DataFrame()
#with jsonlines.open('output.jsonl', mode='w') as writer:
for i, row in df.iterrows():
    #json = {}
    #metadata = {}
    #metadata['id'] = row['id']
    series = pd.Series({'prompt': row['full_text'], 'completion': row['sentiment']})
    #json['metadata'] = metadata
    training_df = training_df.append(series, ignore_index=True)
    if i == 100:
        break

training_df.to_csv("/Users/baset/Downloads/training_1_100.csv")


