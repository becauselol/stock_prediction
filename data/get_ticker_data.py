import yfinance as yf

t = yf.Ticker("spy")
df = t.history("1y")
df.to_csv("sample.csv")
