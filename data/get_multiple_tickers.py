import os
import pandas as pd
import yfinance as yf
from fredapi import Fred
import stockstats
import argparse

def main():
	args = parse_args()
	ticker_csv = args.ticker_csv
	period = args.period if args.period else "5y"
	start = args.start if args.start else "2017-01-01"
	end = args.end if args.end else "2022-12-31"

	fred = Fred(api_key=os.environ["FRED_KEY"])

	print("GETTING FRED DATA...")

	interest_rate_df = fred.get_series('EFFR')
	unemployment_rate_df = fred.get_series('UNRATE')
	consumer_sent_df = fred.get_series('UMCSENT')

	interest_rate_df.index = interest_rate_df.index.strftime("%Y-%m-%d")
	unemployment_rate_df.index = unemployment_rate_df.index.strftime("%Y-%m-%d")
	consumer_sent_df.index = consumer_sent_df.index.strftime("%Y-%m-%d")

	interest_rate_df.name = "EFFR"
	unemployment_rate_df.name = "UNRATE"
	consumer_sent_df.name = "UMCSENT"

	print("COMBINING FRED DATA...")

	combined_df = pd.merge(interest_rate_df, unemployment_rate_df, left_index=True, right_index=True, how="outer")
	combined_df = pd.merge(combined_df, consumer_sent_df, left_index=True, right_index=True, how="outer")

	combined_df.index.name = "Date"

	print("GETTING yfinance DATA...")

	usd = yf.Ticker("DX-Y.NYB")
	usd_df = usd.history(start=start, end=end)
	usd_df.index = usd_df.index.strftime("%Y-%m-%d")
	usd_df.columns = usd_df.columns.str.lower()
	usd_df.columns = [i + "_usd" for i in usd_df.columns]

	vix = yf.Ticker("^VIX")
	vix_df = vix.history(start=start, end=end)
	vix_df.index = vix_df.index.strftime("%Y-%m-%d")
	vix_df.columns = vix_df.columns.str.lower()

	print(f"GETTING ${ticker} DATA...")

	t = yf.Ticker(ticker)
	df = t.history(start=start, end=end)
	df.index = df.index.strftime("%Y-%m-%d")
	df.columns = df.columns.str.lower()

	print("GETTING TECHNICAL INDICATORS...")
	
	ss_df = stockstats.wrap(df)
	ss_df[['macd', 'rsi', 'atr']];

	print("COMBINING ALL DATA...")

	yf_combined_df = ss_df.merge(vix_df, on="Date", how="outer", suffixes = (f"_{ticker}", "_vix"))
	yf_combined_df = yf_combined_df.merge(usd_df, on="Date", how="outer", suffixes=("", "_usd"))

	combined_df = combined_df.merge(yf_combined_df, on="Date", how="outer")

	combined_df["EFFR"] = combined_df["EFFR"].ffill()
	combined_df["UNRATE"] = combined_df["UNRATE"].ffill()
	combined_df["UMCSENT"] = combined_df["UMCSENT"].ffill()

	final_data_df = combined_df.dropna()

	print("WRITING TO CSV...")

	final_data_df.to_csv(f"data_{ticker}_{start}_{end}.csv")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker_csv', required=True)
    parser.add_argument('--period', required=False)
    parser.add_argument('--start', required=False)
    parser.add_argument('--end',required=False)
    return parser.parse_args()

if __name__ == "__main__":
    main()