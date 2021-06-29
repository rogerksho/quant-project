from generate import *
import yfinance as yf
import matplotlib.pyplot as plt

# impt vars
TRAINING_PERIOD = "8y"
TRAINING_END = "2021-03-01"

ALPHA_IN = 0.8 # percentage
BETA_IN = 1.2 # percentage

CUTOFF = 25 # percentage

PRINT_EVAL = 0 # bool

def generate_model(ticker, backtest=False):
    # load data and crop
    stock = yf.Ticker(ticker)
    stock_hist = stock.history(period=TRAINING_PERIOD)
    stock_hist = stock_hist.assign(pct_change_close=stock_hist["Close"].pct_change() * 100)

    # partition training/testing
    stock_test = stock_hist.loc[TRAINING_END :]
    stock_train = stock_hist.loc[: TRAINING_END]

    # generate duration dataframes
    alpha_duration = generate_duration_df(stock_train, ALPHA=ALPHA_IN)
    beta_duration = generate_duration_df(stock_train, BETA=BETA_IN)

    # generate training dataframes
    alpha_train_df = generate_training_df(alpha_duration, stock_train)
    beta_train_df = generate_training_df(beta_duration, stock_train)

    # train model objects
    alpha_obj = train_model(alpha_train_df, print_eval=PRINT_EVAL)
    beta_obj = train_model(beta_train_df, print_eval=PRINT_EVAL)

    # store into custom model class
    alpha_model = Model(alpha_duration, alpha_obj)
    beta_model = Model(beta_duration, beta_obj)

    # backtest and return predictions depending on settings
    if backtest:
        print("===== BACKTEST MODE =====")
        test_start_date = np.datetime64(TRAINING_END)
        return(backtest_model(alpha_model, beta_model, stock_hist, test_start_date))

    # if not backtesting, set up for prediction
    last_alpha_date = generate_duration_df(stock_hist, ALPHA=ALPHA_IN).index[-1]
    last_beta_date = generate_duration_df(stock_hist, BETA=BETA_IN).index[-1]
    current_date = np.datetime64('today')

    alpha_model.test_df = generate_test_df(stock_hist, last_alpha_date, current_date)
    beta_model.test_df = generate_test_df(stock_hist, last_beta_date, current_date)

    return(alpha_model, beta_model)


def backtest_model(alpha_model, beta_model, prices_df, test_start_date):
    # generate pct diff column
    prices_df = prices_df.assign(pct_change_close=prices_df["Close"].pct_change() * 100)

    # prediction_df
    prediction_df = pd.DataFrame()

    # find dates from last hit date for alpha/beta closest to training_end
    alpha_last = alpha_model.get_last_event_date_training()
    beta_last = beta_model.get_last_event_date_training()

    # copy test start date for loop
    test_date = test_start_date

    # backtest up to two days ago
    while test_date <= np.datetime64("today") - np.timedelta64(2, 'D'):
        # skip weekends/holidays
        if test_date not in prices_df.index:
            # increment day before continuing
            test_date += np.timedelta64(1, "D")
            continue

        # update last hit date if needed
        current_pct_change = prices_df.loc[test_date]["pct_change_close"]

        if current_pct_change >= ALPHA_IN:
            # print('alpha event')
            alpha_last = test_date
        elif current_pct_change <= -BETA_IN:
            # print('beta event')
            beta_last = test_date

        # assign test_dfs
        alpha_model.test_df = generate_test_df(prices_df, alpha_last, test_date)
        beta_model.test_df = generate_test_df(prices_df, beta_last, test_date)

        alpha_prob = alpha_model.get_prediction()
        beta_prob = beta_model.get_prediction()

        # generate signal
        prob_diff = alpha_prob - beta_prob

        # deal with friday/saturday crossovers
        if test_date + np.timedelta64(1, "D") not in prices_df.index:
            prediction_date = prices_df.loc[test_date + np.timedelta64(1, "D") :].index[0]
        else:
            prediction_date = test_date + np.timedelta64(1, "D")

        # append
        prediction_df = prediction_df.append(pd.Series(prob_diff, name=prediction_date))

        # increment date
        test_date += np.timedelta64(1, "D")

    # return predictions
    return(prediction_df)



if __name__ == "__main__":
    stock = yf.Ticker("SPY")
    stock_hist = stock.history(period=TRAINING_PERIOD)

    prediction_df = generate_model("SPY", backtest=True)

    # init pyplot
    fig, ax = plt.subplots(1)

    prediction_df.plot(style=".", grid=True, ylim=[-100,100], legend=False, ax=ax)

    # plot cutoffs
    ax.axhline(y=0, color='b', linestyle='-')
    ax.axhline(y=CUTOFF, color='g', linestyle='-')
    ax.axhline(y=-CUTOFF, color='r', linestyle='-')

    ax.set_title("survival model validation")
    ax.set_xlabel('date')
    ax.set_ylabel('survival model signals')

    # real market data
    ax2 = ax.twinx()
    ax2.set_ylabel('closing price')

    validation_df = stock_hist[prediction_df.index[0] : prediction_df.index[-1]]["Close"]
    validation_df.plot(grid=False, ax=ax2)

    # print basic stats
    print("total buys: ", pd.DataFrame(prediction_df > CUTOFF).sum()[0])
    print("total sells: ", pd.DataFrame(prediction_df < -CUTOFF).sum()[0])

    # show plot
    plt.show()


    
    