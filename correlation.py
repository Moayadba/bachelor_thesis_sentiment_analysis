import pandas as pd
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
from scipy.signal import correlate
from scipy.signal import correlation_lags
from sklearn import preprocessing

#variables
ticker = "AAPL"
original_df_location = "/Users/baset/Desktop/Kursanis Thesis/Abschlussarbeit Reddit/AAPL.csv"
evaluation_df_location = "/Users/baset/Desktop/Kursanis Thesis/Datasets/complete run/df_APPL_whole_dataset_gpt_babbage_complete_babbage.xlsx"



stock_historical_data = get_historical_stock_data(ticker)

#stock_historical_data = pd.read_csv("/Users/baset/Desktop/Kursanis Thesis/GME_stock_price_graph.csv")

# #set just_date as an index and make the index as long as the range of the data and fill zeros in the rows with missing dates
stock_historical_data_complete_dates = fill_missing_dates(stock_historical_data, col="just_date")

#calculate the moving average of the stock price at the closing.
stock_historical_data_complete_dates['MA_3'] = stock_historical_data_complete_dates['Close'].rolling(window=3).mean()

#posts with sentiment
eval_df = pd.read_excel(evaluation_df_location)

#assign 1 where sentiment is positive, -1 where negative and 0 where neutral.
eval_df['sentiment_score'] = np.select([(eval_df['model_sentiment']=='positive'), (eval_df['model_sentiment']=='negative')], [1, -1], default=0)

origial_df = pd.read_csv(original_df_location)
eval_df_merged = eval_df.merge(origial_df, how="left", on=['id'])


#calculate total sum of sentiment for each day and number of posts and the ratio of positive posts per day
sentiment_by_day = eval_df_merged.groupby(['NY_Day'], as_index=False)['sentiment_score'].sum()
sentiment_count_per_day = eval_df_merged.groupby(['NY_Day'], as_index=False)['model_sentiment'].count()
sentiment_by_day['positive_sentiment_ratio'] = eval_df_merged['sentiment_score'] / sentiment_count_per_day['model_sentiment']
sentiment_by_day = sentiment_by_day.merge(sentiment_count_per_day, how="left", on=['NY_Day'])
sentiment_by_day = sentiment_by_day.rename(columns={'sentiment_score': 'sentiment_score_total', 'model_sentiment': 'sentiment_count'})

sentiment_by_day['sentiment_score_MA_3'] = sentiment_by_day['sentiment_score_total'].rolling(window=3).mean()

sentiment_by_day['NY_Day'] = pd.to_datetime(sentiment_by_day['NY_Day'])

sentiment_by_day_merged = sentiment_by_day.merge(stock_historical_data_complete_dates, how='left' ,left_on='NY_Day', right_on='just_date')

sentiment_by_day_merged['sentiment_score_total_normalized'] = preprocessing.minmax_scale(sentiment_by_day_merged['sentiment_score_total'])
sentiment_by_day_merged['Close_price_normalized'] = preprocessing.minmax_scale(sentiment_by_day_merged['Close'])


pearson_corr = scipy.stats.pearsonr(sentiment_by_day_merged['sentiment_score_total_normalized'], sentiment_by_day_merged['Close_price_normalized'])

cross_corr = correlate(sentiment_by_day_merged['sentiment_score_total_normalized'], sentiment_by_day_merged['Close_price_normalized'], mode="same")
lags = correlation_lags(sentiment_by_day_merged['sentiment_score_total_normalized'].size, sentiment_by_day_merged['Close_price_normalized'].size, mode="same")
lag = lags[np.argmax(cross_corr)]

cross_corr_2 = sm.tsa.stattools.ccf(sentiment_by_day_merged['Close_price_normalized'], sentiment_by_day_merged['sentiment_score_total_normalized'],  adjusted=False)
lags_2 = correlation_lags(sentiment_by_day_merged['sentiment_score_total_normalized'].size, sentiment_by_day_merged['Close_price_normalized'].size, mode="full")
lag_2 = lags[np.argmax(cross_corr_2)]


###################################

###################################

sentiment_by_day_merged_mania = sentiment_by_day_merged.loc[(sentiment_by_day_merged['just_date'] >= "2021-01-01") & (sentiment_by_day_merged['just_date'] <= "2021-06-01")]


pearson_corr_mania = scipy.stats.pearsonr(sentiment_by_day_merged_mania['sentiment_score_total_normalized'], sentiment_by_day_merged_mania['Close_price_normalized'])

cross_corr_mania = correlate(sentiment_by_day_merged_mania['sentiment_score_total_normalized'], sentiment_by_day_merged_mania['Close_price_normalized'], mode="same")
lags_mania = correlation_lags(sentiment_by_day_merged_mania['sentiment_score_total_normalized'].size, sentiment_by_day_merged_mania['Close_price_normalized'].size, mode="same")
lag_mania = lags[np.argmax(cross_corr_mania)]

cross_corr_2_mania = sm.tsa.stattools.ccf(sentiment_by_day_merged_mania['sentiment_score_total_normalized'], sentiment_by_day_merged_mania['Close_price_normalized'],  adjusted=False)
lags_2_mania = correlation_lags(sentiment_by_day_merged_mania['sentiment_score_total_normalized'].size, sentiment_by_day_merged_mania['Close_price_normalized'].size, mode="full")
lag_2_mania = lags_2_mania[np.argmax(cross_corr_2_mania)]

fig,ax = plt.subplots()
# make a plot
ax.plot(sentiment_by_day_merged_mania['just_date'],
        sentiment_by_day_merged_mania['sentiment_score_total_normalized'],
        color="red")
# set x-axis label
ax.set_xlabel("date", fontsize = 14)
# set y-axis label
ax.set_ylabel("sentiment score by day (normalized)",
              color="red",
              fontsize=14)

ax2=ax.twinx()
# make a plot with different y-axis using second axis object
ax2.plot(sentiment_by_day_merged_mania['just_date'], sentiment_by_day_merged_mania['Close_price_normalized'],color="blue")
ax2.set_ylabel("Closing price",color="blue",fontsize=14)
plt.show()
# save the plot as a file
print("here")


# eval_df_merged = eval_df_merged.merge(sentiment_by_day, how='left', on=['NY_Day'])
#
# eval_df_merged['NY_Day'] = pd.to_datetime(eval_df_merged['NY_Day'])
#
# eval_df_merged = eval_df_merged.merge(stock_historical_data_complete_dates, how='left' ,left_on='NY_Day', right_on='just_date')
#
# eval_df_merged_serires = eval_df_merged.set_index('NY_Day')
#
# eval_df_merged_serires = eval_df_merged_serires[2:]
#
# eval_df_merged_serires = eval_df_merged_serires[eval_df_merged_serires['sentiment_score_MA_3'].notna()]
# eval_df_merged_serires = eval_df_merged_serires[eval_df_merged_serires['MA_3'].notna()]
#
# norm_eval_df_merged_series_sentiment_score = np.linalg.norm(eval_df_merged_serires['sentiment_score_MA_3'])
# eval_df_merged_serires['sentiment_score_MA_3_normalized'] = eval_df_merged_serires['sentiment_score_MA_3'] / norm_eval_df_merged_series_sentiment_score
#
# norm_eval_df_merged_serires_3ma = np.linalg.norm(eval_df_merged_serires['MA_3'])
# eval_df_merged_serires['MA_3_normalized'] = eval_df_merged_serires['MA_3'] / norm_eval_df_merged_serires_3ma

# correlation = correlate(eval_df_merged_serires['MA_3'], eval_df_merged_serires['sentiment_score_MA_3'], mode="full")
#
# lags = correlation_lags(eval_df_merged_serires['MA_3'].size, eval_df_merged_serires['sentiment_score_MA_3'].size, mode="full")
# lag = lags[np.argmax(correlation)]
#
# corr = sm.tsa.stattools.ccf(eval_df_merged_serires['MA_3'], eval_df_merged_serires['sentiment_score_MA_3'], adjusted=False)


