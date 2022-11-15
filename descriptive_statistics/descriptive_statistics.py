import os
import pandas as pd
from datetime import datetime as dt
import pandas as pd
import json

import scipy
import statsmodels.api as sm
from scipy import signal
from numpy.random import default_rng
from utils.correlation_utils import get_historical_stock_data, fill_missing_dates
import random
import numpy as np
import pandas as pd
from scipy.signal import correlate
from scipy.signal import correlation_lags
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

from utils.data_prcessing import prepare_df
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.signal import savgol_filter

processed_data_location = "../processed_data/merged_df_AAPL.csv"


df = pd.read_csv(processed_data_location)

posts_num = len(df)

first_date = dt.strptime(min(df["posting_date"]), "%Y-%m-%d")
last_date =  dt.strptime(max(df["posting_date"]), "%Y-%m-%d")
delta = last_date - first_date
time_period_in_days = delta.days


days_with_posts = (df.groupby(['posting_date']).sum()).index.values.tolist()
num_of_days_with_posts = len(df.groupby(['posting_date']).sum())
avg_num_post_per_day = posts_num / time_period_in_days

posts_distribution_over_time_df = df.groupby(['posting_date'])['id'].count().reset_index(name ='number_of_posts')
posts_distribution_over_time_df['posting_date'] = pd.to_datetime(posts_distribution_over_time_df['posting_date'])

biggest_number_of_posts_in_one_day = max(posts_distribution_over_time_df['number_of_posts'])
smallest_number_of_posts_in_one_day = min(posts_distribution_over_time_df['number_of_posts'])

complete_text = ""
df.fillna('', inplace=True)
for i, row in df.iterrows():
    complete_text = complete_text + row['title'] + " " + row['selftext']

complete_text_words = complete_text.split()
complete_text_words_count = len(complete_text_words)

avg_num_of_words_per_post = complete_text_words_count // posts_num


fig,ax = plt.subplots()
# make a plot
ax.plot(posts_distribution_over_time_df['posting_date'],
        posts_distribution_over_time_df['number_of_posts'],
        color="red")
# set x-axis label
ax.set_xlabel("date", fontsize = 14)
# set y-axis label
ax.set_ylabel("number_of_posts",
              color="red",
              fontsize=14)

print("done.")