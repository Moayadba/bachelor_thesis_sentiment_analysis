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


#variables
ticker = "GME"
original_df_location = "/Users/baset/Desktop/Kursanis Thesis/Abschlussarbeit Reddit/GME_v2.csv"
evaluation_df_location = "/Users/baset/Desktop/Kursanis Thesis/Datasets/complete run/df_GME_whole_dataset_gpt_babbage_complete.xlsx"



#stock_historical_data = get_historical_stock_data(ticker)

stock_historical_data = pd.read_csv("/Users/baset/Desktop/Kursanis Thesis/GME_stock_price_graph.csv")

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

eval_df_merged = eval_df_merged.merge(sentiment_by_day, how='left', on=['NY_Day'])

eval_df_merged['NY_Day'] = pd.to_datetime(eval_df_merged['NY_Day'])
eval_df_merged = eval_df_merged.merge(stock_historical_data_complete_dates, how='left' ,left_on='NY_Day', right_on='just_date')


sentiment_by_day['NY_Day'] = pd.to_datetime(sentiment_by_day['NY_Day'])
sentiment_by_day_merged = sentiment_by_day.merge(stock_historical_data_complete_dates, how='left' ,left_on='NY_Day', right_on='just_date')

#moving average of the total sentiment score
eval_df_merged['sentiment_score_MA_3'] = eval_df_merged['sentiment_score_total'].rolling(window=3).mean()

#sentiment_by_day['GME']=spy_vals
# sentiment_by_day['score_ret'] = sentiment_by_day["score_x"].pct_change()
# sentiment_by_day['gme_ret'] = sentiment_by_day["GME"].pct_change()
# correlation = sentiment_by_day['score_ret'].corr(sentiment_by_day['gme_ret'])
#sentiment_by_day.plot(secondary_y='score', figsize=(10, 6))
#sentiment_by_day_seg = sentiment_by_day.loc[sentiment_by_day['model_sentiment'] >= 5]

eval_df_merged_serires = eval_df_merged.set_index('NY_Day')

sentiment_by_day_merged_series = sentiment_by_day_merged.set_index("NY_Day")


null_for_sentiment_score_MA =eval_df_merged_serires.loc[eval_df_merged_serires['sentiment_score_MA_3'].isna()]

null_for_GME_MA =eval_df_merged_serires.loc[eval_df_merged_serires['MA_3'].isna()]


eval_df_merged_serires = eval_df_merged_serires[2:]

eval_df_merged_serires = eval_df_merged_serires[eval_df_merged_serires['sentiment_score_MA_3'].notna()]
eval_df_merged_serires = eval_df_merged_serires[eval_df_merged_serires['MA_3'].notna()]


norm_eval_df_merged_series_sentiment_score = np.linalg.norm(eval_df_merged_serires['sentiment_score_MA_3'])
eval_df_merged_serires['sentiment_score_MA_3_normalized'] = eval_df_merged_serires['sentiment_score_MA_3'] / norm_eval_df_merged_series_sentiment_score

norm_eval_df_merged_serires_3ma = np.linalg.norm(eval_df_merged_serires['MA_3'])
eval_df_merged_serires['MA_3_normalized'] = eval_df_merged_serires['MA_3'] / norm_eval_df_merged_serires_3ma


##############
sentiment_by_day_merged_serires_close = np.linalg.norm(sentiment_by_day_merged_series['Close'])
sentiment_by_day_merged_series['Close_normalized'] = sentiment_by_day_merged_series['Close'] / norm_eval_df_merged_series_sentiment_score

sentiment_by_day_merged_series_sentiment_score = np.linalg.norm(sentiment_by_day_merged_series['sentiment_score_total'])
sentiment_by_day_merged_series['sentiment_score_total_normalized'] = sentiment_by_day_merged_series['sentiment_score_total'] / sentiment_by_day_merged_series_sentiment_score


result = np.correlate(eval_df_merged_serires['sentiment_score_MA_3_normalized'], eval_df_merged_serires['MA_3_normalized'], mode='full')
index_max_result = max(range(len(result)), key=result.__getitem__)

result_by_day = np.correlate(sentiment_by_day_merged_series['sentiment_score_total_normalized'], sentiment_by_day_merged_series['Close_normalized'], mode='full')
index_max_result_by_day = max(range(len(result_by_day)), key=result_by_day.__getitem__)

#calculate cross correlation
scipy.signal.correlation_lags(sentiment_by_day_merged_series['sentiment_score_total_normalized'], sentiment_by_day_merged_series['Close_normalized'], mode='full')

cross_corr_2 = sm.tsa.stattools.ccf(eval_df_merged_serires['sentiment_score_MA_3_normalized'], eval_df_merged_serires['MA_3_normalized'], adjusted=False)
index_max = max(range(len(cross_corr_2)), key=cross_corr_2.__getitem__)

window = 21
order = 2
y_sf = savgol_filter(eval_df_merged['sentiment_score_MA_3'], window, order)
plt.plot(eval_df_merged['NY_Day'], y_sf)
plt.show()

window = 21
order = 2
y_sf = savgol_filter(eval_df_merged['sentiment_score_MA_3'], window, order)
plt.plot(eval_df_merged['NY_Day'], y_sf)
plt.show()

fig,ax = plt.subplots()
# make a plot
ax.plot(eval_df_merged_serires['just_date'],
        eval_df_merged_serires['sentiment_score_MA_3_normalized'],
        color="red")
# set x-axis label
ax.set_xlabel("NY_Day", fontsize = 14)
# set y-axis label
ax.set_ylabel("score_ratio",
              color="red",
              fontsize=14)

ax2=ax.twinx()
# make a plot with different y-axis using second axis object
ax2.plot(eval_df_merged_serires['just_date'], eval_df_merged_serires['MA_3_normalized'],color="blue")
ax2.set_ylabel("GME",color="blue",fontsize=14)
plt.show()
# save the plot as a file
fig.savefig('two_different_y_axis_for_single_python_plot_with_twinx.jpg',
            format='jpeg',
            dpi=100,
            bbox_inches='tight')
print("here")