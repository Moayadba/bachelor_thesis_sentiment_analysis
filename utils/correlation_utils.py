import numpy as np
import pandas as pd
import yfinance as yf

#get stock data from yahoo finance

def get_historical_stock_data(ticker, period="max"):
    ticker_obj = yf.Ticker(ticker)

    hist = ticker_obj.history(period=period)
    hist['date'] = hist.index
    hist['just_date'] = hist['date'].dt.date
    return hist


def fill_missing_dates(df, col, method="ffill"):
    df[col] = pd.to_datetime(df[col])
    df = (df.set_index(col)
            .reindex(pd.date_range(min(df[col]), max(df[col])))
            .rename_axis([col])
            .fillna(method=method)
            .reset_index())
    return df
