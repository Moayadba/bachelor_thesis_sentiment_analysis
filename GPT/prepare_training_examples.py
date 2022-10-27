import jsonlines
import pandas as pd

df = pd.read_csv("/Users/baset/Desktop/final_training_sample_APPL.csv")
training_df = pd.DataFrame()
prompt_ending = "-> \n\n###\n\n "
#with jsonlines.open('output.jsonl', mode='w') as writer:
for i, row in df.iterrows():
    #json = {}
    #metadata = {}
    prompt_limited = row['full_text'][:2035] if len(row['full_text']) > 2049 else row['full_text']
    prompt = prompt_limited + prompt_ending
    series = pd.Series({'prompt': prompt, 'completion': row['sentiment'].lower()})
    #json['metadata'] = metadata
    training_df = training_df.append(series, ignore_index=True)

training_df.to_csv("/Users/baset/Downloads/training_GME_complete_300_limited.csv")


