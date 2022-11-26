# Sentiment Analysis repository for the bachelor thesis of (Muaid Baset)

This repository contains the complete code and datasets used in the bachelor thesis with the title:  
<strong><em>"Sentiment Analysis of social media posts as a tool to predict stock price changes during the MEME stocks trading mania."</em></strong>

## general setup

this setup is necessary for all models usage

### Prepare Development Environment


Create a python virtualenv and activate it:
```
python3 -m venv venv && . venv/bin/activate
```

Install python dependencies:
```
pip install -v -r requirements.txt
```

development setup:
```
python setup.py develop
```

## Descriptive statistics 
run this script to get a report containing some descriptive statistics about the reddit posts dataset:
```
python3 descriptive_statistics.py
```

## Usage of the Lexicon model 

the existing copy of the LM dictionary was downloaded form [this link](https://sraf.nd.edu/loughranmcdonald-master-dictionary/)

* edited the variables for the input and output files paths in `/sentiment_analysis_all_models/Lexicon/predict_df_lexicon.py`


* run the file for getting predictions with:

```
python3 predict_df_lexicon.py
```

## Usage of the BERT model 

the base code was taken from this  [repo](https://github.com/ProsusAI/finBERT)

the pre-trained model need to be downloaded from [here](https://github.com/ProsusAI/finBERT)

for running the code:

* create a directory for the model under the path: `sentiment_analysis_all_models/finBERT/models/sentiment/<model directory name>`

* Download the model and put it into the directory you just created.
* Put a copy of `config.json` in this same directory. 
* edited the variables for the input and output files paths and the model file path in `/sentiment_analysis_all_models/finBERT/scripts/predict_df_bert.py`


* run the file for getting predictions with:

```
python3 predict_df_bert.py
```

## Usage of the GPT-3 model

to use GPT-3 from Open AI you need to do the following steps: 

1- create an account at [openAI](https://beta.openai.com/docs/introduction)

2- Set your `OPENAI_API_KEY` environment variable, execute the following command in terminal:
```
export OPENAI_API_KEY="<OPENAI_API_KEY>"
```

3- create a file in the root directory and call it: `.env`

4-paste the API Key and the ORG ID in this file, the file should look like this:
```
OPENAI_API_KEY=XXXX
ORG_ID=XXXX
```
5-prepare your training dataset by using the script:

`GPT-3/prepare_training_examples.py`

6- convert the .csv file to .json format using OpenAI CLI tool.
To do so, execute the following command in terminal:
```
openai tools fine_tunes.prepare_data -f <your training .csv file after preperation>
```
7-use OpenAI CLI tool to create a fine-tuned model, by executing the following command in terminal:
```
openai api fine_tunes.create -t <TRAIN_FILE_ID_OR_PATH> -m <BASE_MODEL>
```

<BASE_MODEL> is the model you want to fine-tune, OpenAI offers 4 options here, these options are:
`ada`, `babbage`, `curie`, or `davinci`. to read more about these models click [here](https://beta.openai.com/docs/models/overview)

8- after the fine-tuning is done, copy the ID of the fine-tuned model and paste it in this script:  
`GPT-3/predict_df_gpt3.py`

9- run the script to generate the predictions:

```
python3 predict_df_gpt3.py
```

## Model validation

to test the performance of the model, run the following script after editing the path of the file with the real, and predicted labels:
```
python3 evaluate_model.py
```
the script will print out the confusion matrix and the classification report

## Study the correlation

to calculate the correlation for the complete time period and the pre,intra and post-mania period, run the following script:
```
python3 correlation.py
```