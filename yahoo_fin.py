import yfinance as yf

GME = yf.Ticker("GME")

hist = GME.history(period="max")
hist['date'] = hist.index
hist['just_date'] = hist['date'].dt.date
print('here')