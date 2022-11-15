import pandas as pd
import re
from utils.correlation_utils import fill_missing_dates
from utils.correlation_utils import get_historical_stock_data

# variables
original_reddit_file_location = "../row_data/GME_v2_row.csv"
original_stock_data_location = "../row_data/GME_stock_price_graph.csv"
stock_ticker = "GME"


def fill_missing_dates(df, col, method="ffill"):
    df[col] = pd.to_datetime(df[col])
    df = (df.set_index(col)
            .reindex(pd.date_range(min(df[col]), max(df[col])))
            .rename_axis([col])
            .fillna(method=method)
            .reset_index())
    return df


reddit_df = pd.read_csv(original_reddit_file_location)
reddit_df['posting_date'] = pd.to_datetime(reddit_df['posting_date'])


yahoo_df = pd.read_csv(original_stock_data_location)
yahoo_df = yahoo_df[['Open', 'High', 'Low', 'Close', 'just_date']]
yahoo_df = fill_missing_dates(yahoo_df, col="just_date")

merged_df = reddit_df.merge(yahoo_df,how="left" ,left_on="posting_date", right_on="just_date")
merged_df.to_csv("../processed_data/merged_df_{}.csv".format(stock_ticker))

print("done.")

