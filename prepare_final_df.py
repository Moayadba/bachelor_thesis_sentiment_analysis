import json

import scipy
from sklearn.preprocessing import StandardScaler

from utils.data_prcessing import prepare_df
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.signal import savgol_filter
import statsmodels.api as sm
from scipy import signal
from numpy.random import default_rng
from utils.correlation_utils import get_historical_stock_data, fill_missing_dates
import random
import datetime as dt
import numpy as np
import pandas as pd


df_GME = pd.read_csv('/Users/baset/Desktop/Kursanis Thesis/Abschlussarbeit Reddit/GME_v2.csv')

eval_df_1 = pd.read_excel("/Users/baset/Desktop/Kursanis Thesis/classfied/df_GME_1_10692.xlsx")
eval_df_2 = pd.read_excel("/Users/baset/Desktop/Kursanis Thesis/classfied/df_GME_10692_34566.xlsx")
eval_df_3 = pd.read_excel("/Users/baset/Desktop/Kursanis Thesis/classfied/df_GME_34566_79949.xlsx")

eval_df = pd.concat([eval_df_1, eval_df_2, eval_df_3])

#assign 1 where sentiment is positive, -1 where negative and 0 where neutral.
#eval_df['sentiment_score'] = np.select([(eval_df['model_sentiment']=='positive'), (eval_df['model_sentiment']=='negative')], [1, -1], default=0)

eval_df_merged = eval_df.drop_duplicates(subset='id', keep="last")


eval_df_merged.to_excel("/Users/baset/Desktop/Kursanis Thesis/Datasets/complete run/df_GME_whole_dataset_gpt_babbage_complete.xlsx")