from survival import *
import yfinance as yf

# sector indices to analyze
sector_indices = {
    'technology': 'IYW',
    'computer_tech': '^XCI',
    'semiconductor': 'SOXX',
    'banking': 'KBE',
    'healthcare': 'IYH',
    'consumer_staples': 'XLP',
    'utilities': 'IDU',
}

# model array
model_dict = dict()
prediction_dict = dict()

# validate tickers
for index in sector_indices.items():
    ticker = index[1]
    yf_ticker = yf.Ticker(ticker)

    if yf_ticker.history().empty:
        raise Exception(f'Invalid symbol "{ticker}", no data to be found from yfinance.')

    # generate models
    else:
        alpha_model, beta_model = generate_model(index[1])
        model_dict[index[0]] = (alpha_model, beta_model)

# fetch predictions
for index_model in model_dict.items():
    prediction_dict[index_model[0]] = round(index_model[1][0].get_prediction() - index_model[1][1].get_prediction(), 3)

# sort sectors based on model predictions
prediction_dict_sorted = dict(sorted(prediction_dict.items(), key=lambda item: item[1], reverse=True))

# pretty print
place = 1
print("================================")
for item in prediction_dict_sorted.items():
    print(place, item[0], sector_indices[item[0]], item[1], sep=' | ')
    place += 1

