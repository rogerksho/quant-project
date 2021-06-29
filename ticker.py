import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# sector indices to analyze
sector_indices = {
    'technology': 'XLK',
    'consumer_disc': 'XLY',
    'industrials': 'DIA',
    'energy': 'XLE',
    'finance': 'XLF',
    'materials': 'XLB',
    'real_estate': 'VNQ',
    'healthcare': 'XLV',
    'consumer_staples': 'XLP',
    'utilities': 'XLU',
    'telecom': 'IYZ',
}

joined_tickers = ' '.join(list(sector_indices.values()))
print(joined_tickers)

# download data
# spy = yf.Ticker("SPY")
# data = spy.history(period="10y")
data = yf.download(
    tickers = joined_tickers,
    period = "18y",
    group_by = "ticker",
    auto_adjust = False,
)

print(data['IYZ'])

#plt.show()