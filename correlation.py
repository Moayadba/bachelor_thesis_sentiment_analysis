import scipy
from utils.correlation_utils import get_historical_stock_data, fill_missing_dates
import pandas as pd
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt

#variables
# TICKER is eiter "GME" or "AAPL"
TICKER = "GME"
ORIGINAL_DATA_FILE_PATH = "/Users/baset/PycharmProjects/bachelor_thesis_sentiment_analysis/row_data/GME_row.csv"
PREDICTED_SENTIMENTS_FILE_PATH = "/Users/baset/PycharmProjects/bachelor_thesis_sentiment_analysis/predicted_sentiments_data/GME_complete_dataset_predicted_with_gpt_babbage.xlsx"
STOCK_PRICES_FILE_PATH = "/Users/baset/PycharmProjects/bachelor_thesis_sentiment_analysis/stock_prices_data/GME_stock_price_graph.csv"


#stock_historical_data = get_historical_stock_data(TICKER)

stock_historical_data = pd.read_csv(STOCK_PRICES_FILE_PATH)

# #set just_date as an index and make the index as long as the range of the data and fill zeros in the rows with missing dates
stock_historical_data_complete_dates = fill_missing_dates(stock_historical_data, col="just_date")

#calculate the moving average of the stock price at the closing.
stock_historical_data_complete_dates['MA_3'] = stock_historical_data_complete_dates['Close'].rolling(window=3).mean()

#posts with sentiment
eval_df = pd.read_excel(PREDICTED_SENTIMENTS_FILE_PATH)

#assign 1 where sentiment is positive, -1 where negative and 0 where neutral.
eval_df['sentiment_score'] = np.select([(eval_df['model_sentiment']=='positive'), (eval_df['model_sentiment']=='negative')], [1, -1], default=0)

origial_df = pd.read_csv(ORIGINAL_DATA_FILE_PATH)
eval_df_merged = eval_df.merge(origial_df, how="left", on=['id'])
eval_df_merged = eval_df_merged[['id', 'full_text', 'text_processed', 'text_cleaned', 'model_sentiment', 'sentiment_score', 'Ticker', 'author', 'num_comments', 'score', 'selftext', 'title','posting_date']]


#calculate total sum of sentiment for each day and number of posts and the ratio of positive posts per day
sentiment_by_day = eval_df_merged.groupby(['posting_date'], as_index=False)['sentiment_score'].sum()
sentiment_count_per_day = eval_df_merged.groupby(['posting_date'], as_index=False)['model_sentiment'].count()
sentiment_by_day['positive_sentiment_ratio'] = eval_df_merged['sentiment_score'] / sentiment_count_per_day['model_sentiment']
sentiment_by_day = sentiment_by_day.merge(sentiment_count_per_day, how="left", on=['posting_date'])
sentiment_by_day = sentiment_by_day.rename(columns={'sentiment_score': 'daily_sentiment_score', 'model_sentiment': 'sentiment_count'})

sentiment_by_day['sentiment_score_MA_3'] = sentiment_by_day['daily_sentiment_score'].rolling(window=3).mean()

# only consider posts published after "2020-01-01"
sentiment_by_day = sentiment_by_day.loc[(sentiment_by_day['posting_date'] >= "2020-01-01")]
sentiment_by_day['posting_date'] = pd.to_datetime(sentiment_by_day['posting_date'])


stock_historical_data_complete_dates = stock_historical_data_complete_dates.loc[(stock_historical_data_complete_dates['just_date'] >= "2020-01-01") & (stock_historical_data_complete_dates['just_date'] <= "2022-01-31")]
sentiment_by_day_merged = sentiment_by_day.merge(stock_historical_data_complete_dates, how='left' ,left_on='posting_date', right_on='just_date')

# Normalize the daily sentiment score and the closing price
sentiment_by_day_merged['daily_sentiment_score_normalized'] = preprocessing.minmax_scale(sentiment_by_day_merged['daily_sentiment_score'])
sentiment_by_day_merged['closing_price_normalized'] = preprocessing.minmax_scale(sentiment_by_day_merged['Close'])

########################################
# correlation over complete time period
########################################

pearson_corr_total = scipy.stats.pearsonr(sentiment_by_day_merged['daily_sentiment_score_normalized'], sentiment_by_day_merged['closing_price_normalized'])

print("\n----------------------------------\n")
print('results for the complete time period:')
print('Pearson correlation coefficient value : {}'.format(pearson_corr_total.statistic))
print('p-value : {}'.format(pearson_corr_total.pvalue))

########################################
# correlation during pre-mania period
########################################

sentiment_by_day_merged_pre_mania = sentiment_by_day_merged.loc[(sentiment_by_day_merged['just_date'] >= "2020-01-01") & (sentiment_by_day_merged['just_date'] <= "2020-12-31")]


pearson_corr_pre_mania = scipy.stats.pearsonr(sentiment_by_day_merged_pre_mania['daily_sentiment_score_normalized'], sentiment_by_day_merged_pre_mania['closing_price_normalized'])
print("\n----------------------------------\n")
print('results for the pre-mania period:')
print('Pearson correlation coefficient value : {}'.format(pearson_corr_pre_mania.statistic))
print('p-value : {}'.format(pearson_corr_pre_mania.pvalue))

########################################
# correlation during intra-mania period
########################################

sentiment_by_day_merged_mania = sentiment_by_day_merged.loc[(sentiment_by_day_merged['just_date'] >= "2021-01-01") & (sentiment_by_day_merged['just_date'] <= "2021-02-28")]


pearson_corr_mania = scipy.stats.pearsonr(sentiment_by_day_merged_mania['daily_sentiment_score_normalized'], sentiment_by_day_merged_mania['closing_price_normalized'])

print("\n----------------------------------\n")
print('results for the mania time period:')
print('Pearson correlation coefficient value : {}'.format(pearson_corr_mania.statistic))
print('p-value : {}'.format(pearson_corr_mania.pvalue))

########################################
# correlation during post-mania period
########################################

sentiment_by_day_merged_post_mania = sentiment_by_day_merged.loc[(sentiment_by_day_merged['just_date'] > "2021-02-28")]


pearson_corr_post_mania = scipy.stats.pearsonr(sentiment_by_day_merged_post_mania['daily_sentiment_score_normalized'], sentiment_by_day_merged_post_mania['closing_price_normalized'])
print("\n----------------------------------\n")
print('results for the post-mania time period:')
print('Pearson correlation coefficient value : {}'.format(pearson_corr_post_mania.statistic))
print('p-value : {}'.format(pearson_corr_post_mania.pvalue))

###################################################
# plot daily sentiment score against closing price
###################################################

fig,ax = plt.subplots(figsize=(8,5))
# make a plot
ax.plot(sentiment_by_day_merged['just_date'],
        sentiment_by_day_merged['daily_sentiment_score_normalized'],
        color="red")
# set x-axis label
ax.set_xlabel("date", fontsize = 14)
# set y-axis label
ax.set_ylabel("sentiment score by day (normalized)",
              color="red",
              fontsize=14)

ax2=ax.twinx()
# make a plot with different y-axis using second axis object
ax2.plot(sentiment_by_day_merged['just_date'], sentiment_by_day_merged['closing_price_normalized'],color="blue")
ax2.set_ylabel("Closing price of the stock (normalized)",color="blue",fontsize=14)
plt.show()


