import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sector_helper import *

from ta import momentum, volatility, trend

# disable warnings
pd.options.mode.chained_assignment = None  # default='warn'

# impt vars
BACKTEST_PERIOD = "5y"
STARTING_CAPITAL = 100000
USE_PICKLE = False

# sector indices to analyze
sector_indices = {
    'technology': 'XLK',
    'consumer_disc': 'XLY',
    'industrials': 'XLI',
    'energy': 'XLE',
    'finance': 'XLF',
    'materials': 'XLB',
    'real_estate': 'VNQ',
    'healthcare': 'XLV',
    'consumer_staples': 'XLP',
    'utilities': 'XLU',
    'telecom': 'IYZ',
}


# try fetch from yfinance, if not then used the pickled versions
try:
    # use pickle
    if USE_PICKLE:
        raise Exception('>>> using pickle')

    # download SPY data
    spy = yf.Ticker("SPY")
    spy_hist = spy.history(period=BACKTEST_PERIOD)

    # download VIX data
    vix = yf.Ticker("^VIX")
    vix_hist = vix.history(period=BACKTEST_PERIOD)

    # download all tickers
    indices_df = yf.download(
        tickers = " ".join(sector_indices.values()),
        period = BACKTEST_PERIOD,
        group_by = "ticker",
        auto_adjust = False,
    )

    # pickle dataframes
    with open('pickle/spy_df', 'wb') as outf:
        pickle.dump(spy_hist, outf)

    with open('pickle/vix_df', 'wb') as outf:
        pickle.dump(vix_hist, outf)

    with open('pickle/indices_df', 'wb') as outf:
        pickle.dump(indices_df, outf)

# use pickled if cannot fetch fresh
except BaseException:
    print(">>> couldn't connect to yahoo, using pickeled data instead...")

    # load pickle
    with open('pickle/spy_df', 'rb') as inf:
        spy_hist = pickle.load(inf)

    with open('pickle/vix_df', 'rb') as inf:
        vix_hist = pickle.load(inf)

    #with open('pickle/index_df_dict', 'rb') as inf:
    with open('pickle/indices_df', 'rb') as inf:
        indices_df = pickle.load(inf)


#### process downloaded data ####

# dict to hold all dfs
index_df_dict = dict()

# spy_hist 10m sma
spy_hist["10m_SMA"] = spy_hist["Close"].rolling(210).mean()
spy_hist["10m_EMA"] = spy_hist["Close"].ewm(span=210).mean()

spy_hist.dropna(subset=["10m_SMA"], inplace=True)

# vix 1m sma
vix_hist["sma"] = vix_hist["Close"].rolling(21).mean()

for ticker in list(sector_indices.values()):
    # historical data
    hist_df = indices_df[ticker]

    # return
    hist_df["pct_return"] = hist_df["Close"].pct_change()
    hist_df["pct_return_sma"] = hist_df["pct_return"].ewm(span=63).mean()

    # moving average
    hist_df["rolling_avg_diff"] = (hist_df["Close"].ewm(span=21).mean() - hist_df["Close"].rolling(210).mean()) / hist_df["Close"].rolling(210).mean()

    # momentum measures
    hist_df["MACD"] = trend.MACD(hist_df["Close"]).macd_diff()
    hist_df["MACD_SMA"] = hist_df["MACD"].rolling(168).mean()

    hist_df["RSI"] = momentum.RSIIndicator(hist_df["Close"]).rsi()
    hist_df["RSI_SMA"] = hist_df["RSI"].rolling(21).mean()

    hist_df["stoch_RSI"] = momentum.StochRSIIndicator(hist_df["Close"]).stochrsi_d()

    # volatility measures
    hist_df["return"] = hist_df["Close"].pct_change()

    hist_df["std"] = hist_df["return"].rolling(100).std()
    hist_df["std_sma"] = hist_df["std"].rolling(100).mean()


    # insert into dict
    index_df_dict[ticker] = hist_df


##### TESTING #####
test_start_date = spy_hist.index[0]

# init portfolio
portfolio = Portfolio(index_df_dict, spy_hist, vix_hist, starting_capital=STARTING_CAPITAL,)
portfolio.init_benchmark_spy(test_start_date)

last_rebalance_date = test_start_date

# track prediction accuracy
hit_rate = 0.0
non_empty_holdings = 0.0

# loop
while test_start_date < np.datetime64('today') - np.timedelta64(1, 'D'):
    # adjust for weekends
    if test_start_date not in spy_hist.index:
        if spy_hist.loc[test_start_date :].shape[0] == 0:
            print("backtest completed.")
            break
        test_start_date = spy_hist.loc[test_start_date :].index[0]
        continue

    # record asset value
    portfolio.record_asset_value(test_start_date)
    #print("current asset value:", portfolio.get_asset_value(test_start_date))

    try:
        # this will trip once every end of month
        if not test_start_date.month == spy_hist.iloc[spy_hist.index.get_loc(test_start_date) + 1].name.month:
            # rebalance
            current_holdings = list(portfolio.holdings)
            portfolio.rebalance(spy_hist.iloc[spy_hist.index.get_loc(test_start_date)].name)

            winners = portfolio.get_past_winners(test_start_date)

            sample = 0
            correct = 0
            for ix in current_holdings:
                sample += 1
                if ix in winners:
                    correct += 1

            try:
                print("win%:", correct/sample)
                hit_rate += correct/sample
                non_empty_holdings += 1
            except ZeroDivisionError:
                pass

            print(current_holdings, winners)

    # catch exception from most recent month
    except IndexError:
        pass

    # increment days
    test_start_date += np.timedelta64(1, 'D')

print("max drawdown:", f"{100*portfolio.max_drawdown[1]} %,", portfolio.max_drawdown[0])
print("sharpe ratio:", portfolio.sharpe_ratio())
print("correct%:", hit_rate/non_empty_holdings)

portfolio.monthly_performance().to_csv("monthly_performance.csv")
portfolio.plot_performance()
