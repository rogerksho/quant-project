import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# load data
spy = yf.Ticker("SPY")
hist_spy = spy.history(period="3y")["Close"]

vix = pd.read_csv('data/VIX_History.csv', index_col=0, usecols=[0,4], parse_dates=True)

copper = yf.Ticker("HG=F")
hist_copper = copper.history(period="3y")["Close"]

nasdaq_cels = pd.read_csv('data/NASDAQ_cels.csv', index_col=0, usecols=[0,1], parse_dates=True)
sp_util = pd.read_excel('data/sp_utilities.xls', index_col=0, skiprows=6, skipfooter=4)

# plotting
fig, axs = plt.subplots(5)

axs[0].plot(sp_util, label='S&P utilities')
axs[1].plot(hist_copper, 'g', label='copper futures')
axs[2].plot(nasdaq_cels.loc['2021-06-04' : '2018-06-04'], 'c', label='NASDAQ cels')
axs[3].plot(hist_spy, 'r', label='SPY')
axs[4].plot(vix.loc['2018-06-04' : '2021-06-04'], 'm', label='VIX')

axs[0].legend()
axs[1].legend()
axs[2].legend()
axs[3].legend()
axs[4].legend()

plt.show()