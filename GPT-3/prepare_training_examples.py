import jsonlines
import pandas as pd

INPUT_FILE_PATH = '/Users/baset/PycharmProjects/sentiment_analysis_all_models/training_data/training_dataset.csv'
# path of the file with posts you want to write the predictions to
OUTPUT_FILE_PATH = '/Users/baset/PycharmProjects/sentiment_analysis_all_models/training_data/training_dataset_prepared.csv'

def prepare_df(df):
    training_df = pd.DataFrame()
    prompt_ending = "-> \n\n###\n\n "
    for i, row in df.iterrows():
        prompt_limited = row['full_text'][:2035] if len(row['full_text']) > 2049 else row['full_text']
        prompt = prompt_limited + prompt_ending
        series = pd.Series({'prompt': prompt, 'completion': row['sentiment'].lower()})
        training_df = training_df.append(series, ignore_index=True)

    training_df.to_csv(OUTPUT_FILE_PATH)

if __name__ == '__main__':
    print('\nstarted preparing training dataframe\n\n')
    df = pd.read_csv(INPUT_FILE_PATH)
    prepare_df(df)
    print('done.')


