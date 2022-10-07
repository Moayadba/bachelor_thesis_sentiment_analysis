"""
Program to provide generic parsing for all files in a user-specified directory.
The program assumes the input files have been scrubbed,
  i.e., HTML, ASCII-encoded binary, and any other embedded document structures that are not
  intended to be analyzed have been deleted from the file.

Dependencies:
    Python:  MOD_Load_MasterDictionary_vxxxx.py
    Data:    LoughranMcDonald_MasterDictionary_XXXX.csv

The program outputs:
   1.  File name
   2.  File size (in bytes)
   3.  Number of words (based on LM_MasterDictionary
   4.  Proportion of positive words (use with care - see LM, JAR 2016)
   5.  Proportion of negative words
   6.  Proportion of uncertainty words
   7.  Proportion of litigious words
   8.  Proportion of modal-strong words
   9.  Proportion of modal-weak words
  10.  Proportion of constraining words (see Bodnaruk, Loughran and McDonald, JFQA 2015)
  11.  Number of alphanumeric characters (a-z, A-Z)
  12.  Number of digits (0-9)
  13.  Number of numbers (collections of digits)
  14.  Average number of syllables
  15.  Average word length
  16.  Vocabulary (see Loughran-McDonald, JF, 2015)

  ND-SRAF
  McDonald 201606 : updated 201803; 202107; 202201
"""

import csv
import glob
import re
import string
import sys
import datetime as dt
import MOD_Load_MasterDictionary_v2022 as LM
import pandas as pd
# User defined directory for files to be parsed
TARGET_FILE = '/Users/baset/Downloads/sample_1.xlsx'
# User defined file pointer to LM dictionary
MASTER_DICTIONARY_FILE = 'Loughran-McDonald_MasterDictionary_1993-2021.csv'
# User defined output file
OUTPUT_FILE = '/Users/baset/Downloads/Parser.csv'
# Setup output
OUTPUT_FIELDS = ['file name', 'file size', 'number of words', '% negative', '% positive',
                 '% uncertainty', '% litigious', '% strong modal', '% weak modal',
                 '% constraining', '# of alphabetic', '# of digits',
                 '# of numbers', 'avg # of syllables per word', 'average word length', 'vocabulary']

lm_dictionary = LM.load_masterdictionary(MASTER_DICTIONARY_FILE, print_flag=True)


def main():
    # file_list = glob.glob(TARGET_FILES)
    # n_files = 0
    # for file in file_list:
    #     n_files += 1
    #     print(f'{n_files:,} : {file}')
    output_df = pd.DataFrame()
    file = TARGET_FILE
    df = pd.read_excel(file)
    for i, row  in df.iterrows():
        text = row['full_text']
        output_data = get_data(text)
        output_data[0] = text
        output_data[1] = len(text)
        output_series = map_column(output_data)
        result = pd.concat([row, output_series], ignore_index=False)
        output_df = output_df.append(result, ignore_index=True)
        if i == 1000:
            break
    output_df.to_csv("/Users/baset/Downloads/sample_1_result.csv")



def get_data(doc):

    vdictionary = dict()
    _odata = [0] * 16
    total_syllables = 0
    word_length = 0
    
    tokens = re.findall('\w+', doc)  # Note that \w+ splits hyphenated words
    for token in tokens:
        if not token.isdigit() and len(token) > 1 and token in lm_dictionary:
            _odata[2] += 1  # word count
            word_length += len(token)
            if token not in vdictionary:
                vdictionary[token] = 1
            if lm_dictionary[token].negative: _odata[3] += 1
            if lm_dictionary[token].positive: _odata[4] += 1
            if lm_dictionary[token].uncertainty: _odata[5] += 1
            if lm_dictionary[token].litigious: _odata[6] += 1
            if lm_dictionary[token].strong_modal: _odata[7] += 1
            if lm_dictionary[token].weak_modal: _odata[8] += 1
            if lm_dictionary[token].constraining: _odata[9] += 1
            total_syllables += lm_dictionary[token].syllables

    _odata[10] = len(re.findall('[A-Z]', doc))
    _odata[11] = len(re.findall('[0-9]', doc))
    # drop punctuation within numbers for number count
    doc = re.sub('(?!=[0-9])(\.|,)(?=[0-9])', '', doc)
    doc = doc.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    _odata[12] = len(re.findall(r'\b[-+\(]?[$€£]?[-+(]?\d+\)?\b', doc))
    _odata[13] = total_syllables / _odata[2] if _odata[2] else 0
    _odata[14] = word_length / _odata[2] if _odata[2] else 0
    _odata[15] = len(vdictionary)
    
    # Convert counts to %
    for i in range(3, 9 + 1):
        if _odata[2]:
            _odata[i] = (_odata[i] / _odata[2]) * 100
        else:
            _odata[i] = 0
    # Vocabulary
        
    return _odata

def map_column(output_data):
    result = pd.Series()
    result['full_text'] = output_data[0]
    result['sentiment'] = output_data[1]
    result['number of words'] = output_data[2]
    result['% negative'] = output_data[3]
    result['% positive'] = output_data[4]
    result['% uncertainty'] = output_data[5]
    result['% litigious'] = output_data[6]
    result['% strong modal'] = output_data[7]
    result['% weak modal'] = output_data[8]
    result['% constraining'] = output_data[9]
    result['# of alphabetic'] = output_data[10]
    result['# of digits'] = output_data[11]
    result['# of numbers'] = output_data[12]
    result['avg # of syllables per word'] = output_data[13]
    result['average word length'] = output_data[14]
    result['vocabulary'] = output_data[15]
    return result

if __name__ == '__main__':
    start = dt.datetime.now()
    print(f'\n\n{start.strftime("%c")}\nPROGRAM NAME: {sys.argv[0]}\n')
    main()
    print(f'\n\nRuntime: {(dt.datetime.now()-start)}')
    print(f'\nNormal termination.\n{dt.datetime.now().strftime("%c")}\n')
