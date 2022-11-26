import csv
import glob
import re
import string
import sys
import datetime as dt
import MOD_Load_MasterDictionary_v2022 as LM
import pandas as pd
from Lexicon.Generic_Parser import get_data, map_column, get_final_sentiment

# path of the file with posts you want to predict the sentiment for
INPUT_FILE_PATH = '/Users/baset/Desktop/Kursanis Thesis/Datasets/validatation/GME_final_validation_sample_equal_dist_3.csv'
# path of the file with posts you want to write the predictions to
OUTPUT_FILE_PATH = '/Users/baset/Downloads/GME_final_validation_sample_equal_dist_prediction_lexicon.csv'
# User defined file pointer to LM dictionary
MASTER_DICTIONARY_FILE = 'Loughran-McDonald_MasterDictionary_1993-2021.csv'
# Setup output
OUTPUT_AVAILABLE_FIELDS = ['file name', 'file size', 'number of words', '% negative', '% positive',
                 '% uncertainty', '% litigious', '% strong modal', '% weak modal',
                 '% constraining', '# of alphabetic', '# of digits',
                 '# of numbers', 'avg # of syllables per word', 'average word length', 'vocabulary']
OUTPUT_FIELDS = ['id', 'full_text', 'text_processed', 'sentiment', 'model_sentiment']
lm_dictionary = LM.load_masterdictionary(MASTER_DICTIONARY_FILE, print_flag=True)

def predict_df(df):
    output_df = pd.DataFrame()
    for i, row in df.iterrows():
        text = row['full_text']
        output_data = get_data(text)
        output_data[0] = text
        output_data[1] = len(text)
        output_series = map_column(output_data)
        model_sentiment = get_final_sentiment(output_series)
        result = pd.concat([row, model_sentiment, output_series], ignore_index=False)
        output_df = output_df.append(result, ignore_index=True)
    output_df = output_df[OUTPUT_FIELDS]

    output_df.to_csv(OUTPUT_FILE_PATH)

if __name__ == '__main__':
    start = dt.datetime.now()
    print(f'\n\n{start.strftime("%c")}\nPROGRAM NAME: {sys.argv[0]}\n')
    df = pd.read_csv(INPUT_FILE_PATH)
    predict_df(df)
    print(f'\n\nRuntime: {(dt.datetime.now()-start)}')
    print(f'\nNormal termination.\n{dt.datetime.now().strftime("%c")}\n')
